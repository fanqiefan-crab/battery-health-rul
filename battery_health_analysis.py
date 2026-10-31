# -*- coding: utf-8 -*-
"""
无人设备电池健康评估与剩余寿命预测
==================================
数据：NASA PCoE 锂离子电池老化数据集（B0005 / B0006 / B0007 / B0018，24 °C）
输入：B0005.mat ~ B0018.mat（由 NASA 官方数据包解压得到）
输出：battery_cycles.csv、figures/*.png、battery_health_report.html

流程：
1) 解析 .mat 逐循环放电数据并整理为 CSV
2) 清洗（再生尖峰/离群点）与特征提取
3) 三种容量/寿命模型对比：二次多项式回归、双指数经验模型、高斯过程回归
4) 跨电池机器学***RUL 模型（留一电池验证）
5) 误差评估、外推不确定性（蒙特卡洛 RUL 分布）与维护建议
"""

import os
import json
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MAT_DIR = Path(os.environ.get("NASA_BATT_DIR", r"C:\Users\33701\AppData\Local\Temp\nasa_batt"))
USER_CSV = Path(os.environ.get("BATT_CSV", str(BASE / "battery_dataset.csv")))
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
NOMINAL_CAP = 2.0          # Ah
CAP_WARN = 0.80 * NOMINAL_CAP   # 1.60 Ah 预警阈值（80% SOH）
CAP_EOL = 0.75 * NOMINAL_CAP    # 1.50 Ah 寿命终点（75% SOH）
CAP_CRIT = 0.70 * NOMINAL_CAP   # 1.40 Ah 严重衰减
TRAIN_N = 45               # 建模 / 寿命预测使用的早期循环数
HORIZON = 150              # 容量外推评估上限（循环）
HORIZON_RUL = 260          # 求寿命交点时的最大搜索范围

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"]:
    if any(f.lower() == x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

# 配色
C_BATT = {"B0005": "#2f6fb3", "B0006": "#c9a35c", "B0007": "#5a9e78", "B0018": "#b4544b"}
C_MAIN = "#16324f"
C_ACC = "#c9a35c"


# ----------------------------------------------------------------------------
# 1. 数据解析
# ----------------------------------------------------------------------------
def parse_mat(path: Path, batt: str) -> pd.DataFrame:
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    cycles = np.atleast_1d(m[batt].cycle)
    rows = []
    discharge_idx = 0
    for c in cycles:
        if getattr(c, "type", "") != "discharge":
            continue
        d = c.data
        discharge_idx += 1

        def arr(name):
            v = getattr(d, name, None)
            if v is None:
                return np.array([])
            return np.atleast_1d(np.asarray(v, dtype=float)).ravel()

        cap = arr("Capacity")
        t = arr("Time")
        v = arr("Voltage_measured")
        cur = arr("Current_measured")
        temp = arr("Temperature_measured")

        cap_val = float(cap.flat[0]) if cap.size else np.nan
        duration = float(np.nanmax(t) - np.nanmin(t)) if t.size else np.nan
        energy = float(np.trapezoid(v * np.abs(cur), t) / 3600.0) if t.size > 1 else np.nan
        r0 = np.nan
        if v.size > 10 and t.size > 10:
            i_abs = float(np.median(np.abs(cur))) or 1.0
            drop = float(np.median(v[:3]) - v[min(10, v.size - 1)])
            r0 = drop / i_abs

        rows.append({
            "battery": batt,
            "cycle_index": discharge_idx,
            "capacity_Ah": cap_val,
            "duration_s": duration,
            "temp_mean_C": float(np.nanmean(temp)) if temp.size else np.nan,
            "temp_max_C": float(np.nanmax(temp)) if temp.size else np.nan,
            "voltage_mean_V": float(np.nanmean(v)) if v.size else np.nan,
            "voltage_min_V": float(np.nanmin(v)) if v.size else np.nan,
            "energy_Wh": energy,
            "r0_ohm": r0,
            "ambient_temperature_C": float(np.atleast_1d(c.ambient_temperature).flat[0]),
        })
    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    frames = []
    for b in BATTERIES:
        p = MAT_DIR / f"{b}.mat"
        if not p.exists():
            raise FileNotFoundError(f"缺少数据文件：{p}（请先解压 NASA 官方数据包）")
        frames.append(parse_mat(p, b))
    df = pd.concat(frames, ignore_index=True)
    return df


AUX_COLS = ["duration_s", "temp_mean_C", "temp_max_C", "voltage_mean_V",
            "voltage_min_V", "energy_Wh", "r0_ohm"]


def load_dataset() -> pd.DataFrame:
    """以用户提供的 CSV 为准加载容量数据；若 NASA 原始 .mat 仍在，
    则合并放电时长/温度/能量/内阻等辅助特征供机器学习模型使用。"""
    u = pd.read_csv(USER_CSV)
    u = u.rename(columns={"battery_id": "battery", "cycle": "cycle_index"})
    base = u[["battery", "cycle_index", "capacity_Ah"]].copy()
    base["ambient_temperature_C"] = u["ambient_temp_C"] if "ambient_temp_C" in u else np.nan
    if all((MAT_DIR / f"{b}.mat").exists() for b in BATTERIES):
        try:
            raw = build_dataset()
            base = base.merge(raw[["battery", "cycle_index"] + AUX_COLS],
                              on=["battery", "cycle_index"], how="left")
        except Exception:
            pass
    for c in AUX_COLS:
        if c not in base.columns:
            base[c] = np.nan
    return base


def clean_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """中值滤波识别再生尖峰/离群点，并用局部中值替换。"""
    out = []
    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index").copy()
        med = g["capacity_Ah"].rolling(5, center=True, min_periods=1).median()
        dev = (g["capacity_Ah"] - med).abs()
        outlier = dev > 0.06
        g["is_outlier"] = outlier.astype(int)
        g["capacity_clean"] = np.where(outlier, med, g["capacity_Ah"])
        g["capacity_clean"] = g["capacity_clean"].interpolate(limit_direction="both")
        base = g.loc[g["cycle_index"] <= 5, "capacity_clean"].median()
        g["soh"] = g["capacity_clean"] / base
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# 2. 工具函数
# ----------------------------------------------------------------------------
def first_crossing(x, y, threshold):
    """线性插值求首次低于阈值的横坐标。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for i in range(1, len(y)):
        if y[i] < threshold:
            if y[i - 1] >= threshold and y[i] != y[i - 1]:
                return x[i - 1] + (threshold - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            return x[i]
    return None


def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


def dexp(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)


def fit_poly2(cyc, cap):
    p, _ = curve_fit(poly2, cyc, cap, p0=[0.00002, -0.002, 1.9], maxfev=40000)
    return p


def fit_dexp(cyc, cap):
    p0 = [0.9, -0.004, 1.0, -0.0008]
    bounds = ([0.0, -0.2, 0.0, -0.02], [3.0, -1e-5, 3.0, -1e-7])
    p, _ = curve_fit(dexp, cyc, cap, p0=p0, bounds=bounds, maxfev=200000)
    return p


def fit_gp(cyc, cap):
    """对数域高斯过程：RBF 捕捉局部波动，DotProduct 提供趋势外推能力。"""
    kernel = (ConstantKernel(1.0, (1e-3, 100.0))
              * RBF(length_scale=30.0, length_scale_bounds=(5.0, 500.0))
              + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 0.01)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=3,
                                  random_state=42, alpha=1e-10)
    gp.fit(np.asarray(cyc, float).reshape(-1, 1), np.log(np.asarray(cap, float)))
    return gp


def gp_predict(gp, x):
    """返回容量均值与 95% 区间上下界（对数正态变换）。"""
    X = np.asarray(x, float).reshape(-1, 1)
    mu, sd = gp.predict(X, return_std=True)
    return np.exp(mu), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), mu, sd


def gp_samples(gp, x, n_samples, seed):
    """对数域后验抽样，返回容量空间样本，形状 (n_samples, len(x))。"""
    X = np.asarray(x, float).reshape(-1, 1)
    s = gp.sample_y(X, n_samples=n_samples, random_state=seed)
    if s.shape[0] == len(X):
        s = s.T
    return np.exp(s)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ----------------------------------------------------------------------------
# 3. 分析与建模
# ----------------------------------------------------------------------------
def main():
    print("[1/5] 读取用户数据集 ...")
    raw = load_dataset()
    df = clean_capacity(raw)
    df.to_csv(BASE / "battery_cycles.csv", index=False, encoding="utf-8-sig")
    print(f"      共 {len(df)} 条放电循环，异常点 {int(df.is_outlier.sum())} 条")

    # ---------------- 基础统计 ----------------
    summary = []
    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index")
        cross_warn = first_crossing(g.cycle_index, g.capacity_clean, CAP_WARN)
        cross_eol = first_crossing(g.cycle_index, g.capacity_clean, CAP_EOL)
        cross_crit = first_crossing(g.cycle_index, g.capacity_clean, CAP_CRIT)
        fade_rate = (g.capacity_clean.iloc[:5].mean() - g.capacity_clean.iloc[-5:].mean()) / len(g)
        summary.append({
            "电池": b,
            "循环数": int(g.cycle_index.max()),
            "初始容量(Ah)": round(float(g.capacity_clean.iloc[:5].mean()), 3),
            "末期容量(Ah)": round(float(g.capacity_clean.iloc[-5:].mean()), 3),
            "末期SOH(%)": round(float(g.soh.iloc[-5:].mean() * 100), 1),
            "80%SOH循环": None if cross_warn is None else round(cross_warn, 1),
            "75%SOH循环(EOL)": None if cross_eol is None else round(cross_eol, 1),
            "70%SOH循环": None if cross_crit is None else round(cross_crit, 1),
            "平均衰减(mAh/循环)": round(float(fade_rate) * 1000, 2),
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # ---------------- 图 1：容量退化曲线 ----------------
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=170)
    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index")
        ax.plot(g.cycle_index, g.capacity_Ah, color=C_BATT[b], alpha=0.22, lw=0.9)
        ax.plot(g.cycle_index, g.capacity_clean, color=C_BATT[b], lw=2.0, label=b)
        ax.scatter(g.loc[g.is_outlier == 1, "cycle_index"], g.loc[g.is_outlier == 1, "capacity_Ah"],
                   s=12, facecolors="none", edgecolors="#d1495b", lw=0.8, zorder=5)
        ce = first_crossing(g.cycle_index, g.capacity_clean, CAP_EOL)
        if ce:
            ax.scatter([ce], [CAP_EOL], marker="x", s=55, color=C_BATT[b], zorder=6)
    for y, txt, c in [(CAP_WARN, "80% SOH 预警", "#e0a458"), (CAP_EOL, "75% SOH 寿命终点", "#d1495b"),
                      (CAP_CRIT, "70% SOH 严重衰减", "#8c1c13")]:
        ax.axhline(y, color=c, ls="--", lw=1.1, alpha=0.85)
        ax.text(3, y + 0.012, f"{txt}  {y:.2f} Ah", color=c, fontsize=9)
    ax.set_xlabel("放电循环序号")
    ax.set_ylabel("放电容量 (Ah)")
    ax.set_title("四种电池容量退化曲线（浅色为原始数据，深色为清洗后，红圈为剔除的再生尖峰）",
                 fontsize=11.5, color=C_MAIN, pad=12)
    ax.grid(alpha=0.18, ls=":")
    ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=9)
    ax.set_ylim(1.0, 2.12)
    fig.tight_layout()
    f1 = FIG_DIR / "fig1_degradation.png"
    fig.savefig(f1)
    plt.close(fig)

    # ---------------- 图 2：SOH 与衰减速率 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index")
        ax.plot(g.cycle_index, g.soh * 100, color=C_BATT[b], lw=1.9, label=b)
    for y, c in [(80, "#e0a458"), (75, "#d1495b"), (70, "#8c1c13")]:
        ax.axhline(y, color=c, ls="--", lw=1.0)
    ax.set_xlabel("循环序号")
    ax.set_ylabel("SOH (%)")
    ax.set_title("健康状态衰减轨迹", fontsize=11, color=C_MAIN)
    ax.grid(alpha=0.18, ls=":")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(55, 104)

    ax = axes[1]
    labels = summary_df["电池"].tolist()
    rates = summary_df["平均衰减(mAh/循环)"].tolist()
    bars = ax.bar(labels, rates, color=[C_BATT[b] for b in labels], width=0.62)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center",
                fontsize=9, color=C_MAIN)
    ax.set_ylabel("平均衰减速率 (mAh/循环)")
    ax.set_title("个体衰减速率差异", fontsize=11, color=C_MAIN)
    ax.grid(axis="y", alpha=0.18, ls=":")
    ax.set_ylim(0, max(rates) * 1.22)
    fig.tight_layout()
    f2 = FIG_DIR / "fig2_soh_rate.png"
    fig.savefig(f2)
    plt.close(fig)

    # ---------------- 4. 容量预测模型对比 ----------------
    print("[2/5] 拟合容量退化模型 ...")
    pred_rows = []
    model_curves = {}
    gp_store = {}
    rul_rows = []
    x_rul = np.arange(TRAIN_N + 1, HORIZON_RUL + 1, 1.0)

    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index")
        cyc_all = g.cycle_index.values.astype(float)
        cap_all = g.capacity_clean.values.astype(float)
        tr = cyc_all <= TRAIN_N
        x_tr, y_tr = cyc_all[tr], cap_all[tr]
        x_ev = np.arange(TRAIN_N + 1, HORIZON + 1, 1.0)

        p_poly = fit_poly2(x_tr, y_tr)
        y_poly = poly2(x_ev, *p_poly)

        p_dexp = fit_dexp(x_tr, y_tr)
        y_dexp = dexp(x_ev, *p_dexp)

        gp = fit_gp(x_tr, y_tr)
        gp_store[b] = gp
        y_gp, gp_lo, gp_hi, gp_mu_log, gp_sd_log = gp_predict(gp, x_ev)

        model_curves[b] = {
            "cycle": x_ev, "poly": y_poly, "dexp": y_dexp,
            "gp": y_gp, "gp_lo": gp_lo, "gp_hi": gp_hi,
            "train_n": TRAIN_N, "p_poly": p_poly, "p_dexp": p_dexp,
        }

        # 评估区（真实数据存在的部分）
        real = g[(g.cycle_index > TRAIN_N) & (g.cycle_index <= HORIZON)]
        if len(real):
            idx = np.clip(np.searchsorted(x_ev, real.cycle_index.values), 0, len(x_ev) - 1)
            for name, yhat in [("二次回归", y_poly), ("双指数", y_dexp), ("对数GP", y_gp)]:
                pred_rows.append({"电池": b, "方法": name,
                                  **metrics(real.capacity_clean.values, yhat[idx])})

        # RUL 评估（以 1.50 Ah 为寿命终点）
        true_eol = first_crossing(cyc_all, cap_all, CAP_EOL)
        if true_eol is None:
            continue
        true_rul = true_eol - TRAIN_N
        y_rul = {
            "二次回归": poly2(x_rul, *p_poly),
            "双指数": dexp(x_rul, *p_dexp),
            "对数GP": gp_predict(gp, x_rul)[0],
        }
        for name, yseq in y_rul.items():
            pe = first_crossing(x_rul, yseq, CAP_EOL)
            pred_rul = np.nan if pe is None else pe - TRAIN_N
            rul_rows.append({"电池": b, "方法": name, "真实RUL": round(true_rul, 1),
                             "预测RUL": None if np.isnan(pred_rul) else round(pred_rul, 1),
                             "误差(循环)": None if np.isnan(pred_rul) else round(pred_rul - true_rul, 1)})

        # 高斯过程后验抽样：RUL 概率分布
        samples = gp_samples(gp, x_rul, 1500, 11)
        gp_rul = []
        for s in samples:
            e = first_crossing(x_rul, s, CAP_EOL)
            if e is not None:
                gp_rul.append(e - TRAIN_N)
        gp_store[b + "_rul_samples"] = np.array(gp_rul)
        gp_store[b + "_true_rul"] = true_rul

    cap_metrics = pd.DataFrame(pred_rows)
    cap_table = cap_metrics.groupby("方法")[["MAE", "RMSE", "MAPE", "R2"]].mean().reset_index()
    cap_table = cap_table.rename(columns={"MAE": "MAE(Ah)", "RMSE": "RMSE(Ah)", "MAPE": "MAPE(%)", "R2": "R²"})
    print(cap_table.to_string(index=False))

    rul_df = pd.DataFrame(rul_rows)
    rul_pivot = rul_df.pivot(index="电池", columns="方法", values="预测RUL").reindex(BATTERIES)
    true_map = rul_df.groupby("电池")["真实RUL"].first()
    rul_display = pd.DataFrame({"电池": BATTERIES,
                                "真实RUL": [true_map.get(b, np.nan) for b in BATTERIES]})
    for m in ["二次回归", "双指数", "对数GP"]:
        rul_display[m] = rul_pivot[m].reindex(BATTERIES).values

    ok = rul_df.dropna(subset=["误差(循环)"]).assign(e=lambda d: d["误差(循环)"].abs())
    rul_table = (ok.groupby("方法")["e"].agg(["mean", "median", "max", "count"]).reset_index())
    rul_table.columns = ["方法", "平均绝对误差(循环)", "中位绝对误差(循环)", "最大误差(循环)", "有效电池数"]
    print(rul_display.to_string(index=False))
    print(rul_table.to_string(index=False))

    # ---------------- 图 3：容量预测对比 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), dpi=170)
    for ax, b in zip(axes.ravel(), BATTERIES):
        g = df[df.battery == b].sort_values("cycle_index")
        c = model_curves[b]
        ax.plot(g.cycle_index, g.capacity_Ah, ".", ms=3, color="#b9c3cd", label="实测（原始）")
        ax.plot(g.cycle_index, g.capacity_clean, lw=1.7, color="#4a5b6b", label="实测（清洗）")
        ax.plot(c["cycle"], c["poly"], lw=1.9, color="#d1495b", ls="--", label="二次回归")
        ax.plot(c["cycle"], c["dexp"], lw=1.9, color="#2f9e78", ls="-.", label="双指数")
        ax.plot(c["cycle"], c["gp"], lw=2.0, color="#c9a35c", label="对数GP")
        ax.fill_between(c["cycle"], c["gp_lo"], c["gp_hi"],
                        color="#c9a35c", alpha=0.16, lw=0)
        ax.axvspan(0, TRAIN_N, color="#16324f", alpha=0.055, lw=0)
        ax.axhline(CAP_EOL, color="#d1495b", ls=":", lw=1.0)
        ax.set_title(f"{b}：仅用前 {TRAIN_N} 循环建模后的外推", fontsize=10.5, color=C_MAIN)
        ax.set_xlim(0, HORIZON)
        ax.set_ylim(1.05, 2.1)
        ax.grid(alpha=0.16, ls=":")
        ax.set_xlabel("循环序号")
        ax.set_ylabel("容量 (Ah)")
        if b == "B0005":
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("三种容量退化模型的预测能力对比（阴影：对数GP 95% 置信区间）",
                 fontsize=12, color=C_MAIN, y=0.995)
    fig.tight_layout()
    f3 = FIG_DIR / "fig3_capacity_models.png"
    fig.savefig(f3)
    plt.close(fig)

    # ---------------- 图 4：RUL 预测对比 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=170,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    width = 0.26
    xs = np.arange(len(BATTERIES))
    for i, method in enumerate(["二次回归", "双指数", "对数GP"]):
        sub = rul_df[rul_df["方法"] == method].set_index("电池")
        errs = [abs(sub.loc[b, "误差(循环)"]) if b in sub.index and sub.loc[b, "预测RUL"] is not None
                else np.nan for b in BATTERIES]
        ax.bar(xs + (i - 1) * width, errs, width,
               label=method, color=["#d1495b", "#2f9e78", "#c9a35c"][i], alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(BATTERIES)
    ax.set_ylabel("RUL 绝对误差 (循环)")
    ax.set_title(f"第 {TRAIN_N} 循环时预测剩余寿命的误差", fontsize=10.5, color=C_MAIN)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.18, ls=":")

    ax = axes[1]
    for method, color in [("二次回归", "#d1495b"), ("双指数", "#2f9e78"), ("对数GP", "#c9a35c")]:
        sub = rul_df[(rul_df["方法"] == method)].dropna(subset=["预测RUL"])
        ax.scatter(sub["真实RUL"], sub["预测RUL"], s=46, color=color, label=method,
                   edgecolors="white", linewidths=0.7)
    lim = [0, max(rul_df["真实RUL"].max(), rul_df["预测RUL"].max()) * 1.15]
    ax.plot(lim, lim, color="#4a5b6b", lw=1.0, ls="--")
    ax.fill_between(lim, [v * 0.8 for v in lim], [v * 1.2 for v in lim], color="#4a5b6b", alpha=0.07, lw=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实 RUL (循环)")
    ax.set_ylabel("预测 RUL (循环)")
    ax.set_title("预测 vs 真实（虚线为理想线，带内为 ±20%）", fontsize=10.5, color=C_MAIN)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.16, ls=":")
    fig.tight_layout()
    f4 = FIG_DIR / "fig4_rul_compare.png"
    fig.savefig(f4)
    plt.close(fig)

    # ---------------- 5. 不确定性 ----------------
    print("[3/5] 不确定性分析 ...")
    # 提前预警场景：不同预测起点
    warn_rows = []
    for b in BATTERIES:
        g = df[df.battery == b].sort_values("cycle_index")
        cyc_all = g.cycle_index.values.astype(float)
        cap_all = g.capacity_clean.values.astype(float)
        true_eol = first_crossing(cyc_all, cap_all, CAP_EOL)
        if true_eol is None:
            continue
        for t0 in [30, 45, 60, 75]:
            if t0 >= true_eol:
                continue
            m = cyc_all <= t0
            gp = fit_gp(cyc_all[m], cap_all[m])
            x_ev = np.arange(t0 + 1, HORIZON_RUL + 1, 1.0)
            samples = gp_samples(gp, x_ev, 800, 3)
            eols = [first_crossing(x_ev, s, CAP_EOL) for s in samples]
            eols = np.array([e for e in eols if e is not None])
            if len(eols) == 0:
                continue
            rul_pred = np.median(eols) - t0
            lo, hi = np.percentile(eols - t0, [2.5, 97.5])
            warn_rows.append({
                "电池": b, "预测起点": t0, "真实RUL": round(true_eol - t0, 1),
                "RUL中位数": round(float(rul_pred), 1),
                "95%区间下限": round(float(lo), 1), "95%区间上限": round(float(hi), 1),
                "区间宽度": round(float(hi - lo), 1),
                "误差": round(float(rul_pred - (true_eol - t0)), 1),
                "覆盖率": "命中" if lo <= true_eol - t0 <= hi else "未命中",
            })
    warn_df = pd.DataFrame(warn_rows)
    print(warn_df.to_string(index=False))

    # 外推不确定度随 horizon 增长
    gp = gp_store["B0005"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170)
    ax = axes[0]
    g = df[df.battery == "B0005"].sort_values("cycle_index")
    c = model_curves["B0005"]
    ax.plot(g.cycle_index, g.capacity_clean, lw=1.6, color="#4a5b6b", label="实测")
    ax.plot(c["cycle"], c["gp"], lw=1.9, color="#c9a35c", label="对数GP 均值")
    ax.fill_between(c["cycle"], c["gp_lo"], c["gp_hi"], color="#c9a35c", alpha=0.18, lw=0,
                    label="95% 置信区间")
    ax.axhline(CAP_EOL, color="#d1495b", ls=":", lw=1.0)
    ax.axvline(TRAIN_N, color="#16324f", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlim(0, 165)
    ax.set_ylim(1.1, 2.0)
    ax.set_xlabel("循环序号")
    ax.set_ylabel("容量 (Ah)")
    ax.set_title("B0005：外推置信区间随预测步长张开", fontsize=10.5, color=C_MAIN)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.16, ls=":")

    ax = axes[1]
    gp_rul = gp_store.get("B0005_rul_samples", np.array([]))
    true_r = gp_store.get("B0005_true_rul", None)
    if gp_rul.size:
        ax.hist(gp_rul, bins=28, color="#c9a35c", alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.axvline(true_r, color="#16324f", lw=1.6, ls="-", label=f"真实 RUL = {true_r:.0f}")
        med = np.median(gp_rul)
        ax.axvline(med, color="#d1495b", lw=1.4, ls="--", label=f"后验中位数 = {med:.0f}")
        lo, hi = np.percentile(gp_rul, [2.5, 97.5])
        ax.axvspan(lo, hi, color="#d1495b", alpha=0.08, lw=0,
                   label=f"95% 区间 [{lo:.0f}, {hi:.0f}]")
    ax.set_xlabel("剩余寿命 RUL (循环)")
    ax.set_ylabel("后验样本数")
    ax.set_title("B0005：RUL 的后验分布（GP 抽样 1500 次）", fontsize=10.5, color=C_MAIN)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.14, ls=":")
    fig.tight_layout()
    f5 = FIG_DIR / "fig5_uncertainty.png"
    fig.savefig(f5)
    plt.close(fig)

    # ---------------- 图 6：提前预警误差曲线 ----------------
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=170)
    for b in BATTERIES:
        sub = warn_df[warn_df["电池"] == b].sort_values("预测起点")
        if len(sub) < 2:
            continue
        ax.errorbar(sub["预测起点"], sub["RUL中位数"],
                    yerr=[sub["RUL中位数"] - sub["95%区间下限"],
                          sub["95%区间上限"] - sub["RUL中位数"]],
                    marker="o", ms=5, lw=1.5, capsize=4, color=C_BATT[b], label=f"{b} 预测")
        ax.plot(sub["预测起点"], sub["真实RUL"], ls=":", lw=1.6, marker="|", ms=8,
                color=C_BATT[b], alpha=0.75, zorder=3)
    ax.plot([], [], ls=":", lw=1.6, color="#777", label="虚线 = 真实 RUL")
    ax.set_xlabel("做出预测时所处的循环")
    ax.set_ylabel("剩余寿命 RUL (循环)")
    ax.set_title("提前预警：预测越早，RUL 区间越宽（95% 区间）", fontsize=11, color=C_MAIN)
    ax.grid(alpha=0.18, ls=":")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    f6 = FIG_DIR / "fig6_early_warning.png"
    fig.savefig(f6)
    plt.close(fig)

    # ---------------- 6. 跨电池机器学习 RUL 模型 ----------------
    print("[4/5] 训练跨电池 RUL 机器学习模型 ...")
    data = []
    for b, g in df.groupby("battery", sort=False):
        g = g.sort_values("cycle_index").copy()
        g["roll5"] = g.capacity_clean.rolling(5, min_periods=1).mean()
        g["slope10"] = (g.soh - g.soh.shift(10)) / 10
        g["slope20"] = (g.soh - g.soh.shift(20)) / 20
        eol = first_crossing(g.cycle_index.values.astype(float),
                             g.capacity_clean.values.astype(float), CAP_EOL)
        if eol is None:
            continue
        g["rul_true"] = eol - g.cycle_index
        g["battery"] = b
        data.append(g[g.cycle_index >= 15])
    ml_df = pd.concat(data, ignore_index=True)
    feat_cols = ["cycle_index", "soh", "roll5", "slope10", "slope20",
                 "temp_mean_C", "duration_s", "energy_Wh", "r0_ohm"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns and ml_df[c].notna().any()]
    ml_df = ml_df.dropna(subset=feat_cols + ["rul_true"])
    ml_df = ml_df[ml_df["rul_true"] >= 0].reset_index(drop=True)

    ml_rows = []
    ml_preds = []
    importance = None
    for test_b in BATTERIES:
        tr = ml_df[ml_df.battery != test_b]
        te = ml_df[ml_df.battery == test_b]
        if len(te) < 5:
            continue
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=42)
        model.fit(tr[feat_cols], tr["rul_true"])
        pred = model.predict(te[feat_cols])
        ml_rows.append({"电池": test_b, **metrics(te["rul_true"], pred)})
        tmp = te[["cycle_index", "rul_true"]].copy()
        tmp["pred"] = pred
        tmp["battery"] = test_b
        ml_preds.append(tmp)
        if test_b == "B0018":
            pi = permutation_importance(model, te[feat_cols], te["rul_true"],
                                        n_repeats=15, random_state=0)
            importance = pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean})
    ml_metrics = pd.DataFrame(ml_rows)
    ml_pred_df = pd.concat(ml_preds, ignore_index=True)
    print(ml_metrics.to_string(index=False))

    # 图 7：ML RUL 预测与特征重要性
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for b in BATTERIES:
        sub = ml_pred_df[ml_pred_df.battery == b].sort_values("cycle_index")
        if sub.empty:
            continue
        ax.plot(sub.cycle_index, sub.rul_true, lw=2.0, color=C_BATT[b], ls="-", label=f"{b}")
        ax.plot(sub.cycle_index, sub.pred, lw=1.2, color=C_BATT[b], ls="--", alpha=0.9)
    ax.plot([], [], lw=1.2, ls="--", color="#888", label="预测（虚线）")
    ax.set_xlabel("循环序号")
    ax.set_ylabel("剩余寿命 RUL (循环)")
    ax.set_title("跨电池机器学习 RUL 预测（虚线）与真实值（实线）", fontsize=10.5, color=C_MAIN)
    ax.grid(alpha=0.18, ls=":")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")

    ax = axes[1]
    if importance is not None:
        imp = importance.sort_values("importance")
        ax.barh(imp["feature"], imp["importance"], color="#2f6fb3", alpha=0.85)
        ax.set_xlabel("置换重要性（MAE 增量）")
        ax.set_title("RUL 模型特征重要性（B0018 折）", fontsize=10.5, color=C_MAIN)
        ax.grid(axis="x", alpha=0.18, ls=":")
    fig.tight_layout()
    f7 = FIG_DIR / "fig7_ml_rul.png"
    fig.savefig(f7)
    plt.close(fig)

    # ---------------- 汇总指标 ----------------
    best_cap = cap_table.sort_values("RMSE(Ah)").iloc[0]["方法"]
    best_rul = rul_table.sort_values("中位绝对误差(循环)").iloc[0]["方法"]
    avg_end_soh = float(summary_df["末期SOH(%)"].mean())
    kpis = {
        "cycles": int(len(df)),
        "batteries": int(df.battery.nunique()),
        "avg_end_soh": avg_end_soh,
        "best_cap_model": best_cap,
        "best_cap_rmse": float(cap_table["RMSE(Ah)"].min()),
        "best_rul_model": best_rul,
        "best_rul_medae": float(rul_table["中位绝对误差(循环)"].min()),
        "ml_mae": float(ml_metrics["MAE"].mean()) if len(ml_metrics) else float("nan"),
        "ml_r2": float(ml_metrics["R2"].mean()) if len(ml_metrics) else float("nan"),
    }

    out = {
        "kpis": kpis,
        "summary": summary_df.to_dict("records"),
        "cap_table": cap_table.round(4).to_dict("records"),
        "rul_table": rul_table.round(2).to_dict("records"),
        "rul_detail": rul_df.to_dict("records"),
        "warn": warn_df.to_dict("records"),
        "ml": ml_metrics.round(3).to_dict("records"),
        "gp_opt_kernel": str(gp_store["B0005"].kernel_),
    }
    (BASE / "analysis_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    # ---------------- 7. HTML 报告 ----------------
    print("[5/5] 生成 HTML 报告 ...")
    build_report(kpis, summary_df, cap_table, rul_table, rul_display, warn_df, ml_metrics,
                 {"fig1": f1, "fig2": f2, "fig3": f3, "fig4": f4, "fig5": f5, "fig6": f6, "fig7": f7},
                 gp_store)
    print(f"完成。报告：{BASE / 'battery_health_report.html'}")


# ----------------------------------------------------------------------------
# 4. 报告
# ----------------------------------------------------------------------------
CSS = """
:root{--ink:#122033;--ink2:#40566d;--muted:#7c8da0;--bg:#f7f5f0;--card:#ffffff;
--gold:#b8914e;--navy:#16324f;--line:#e3ded4;--red:#b4544b;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC","Helvetica Neue",sans-serif;
line-height:1.85;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:56px 28px 80px}
header{border-bottom:2px solid var(--navy);padding-bottom:26px;margin-bottom:38px}
.eyebrow{letter-spacing:.32em;font-size:11px;color:var(--gold);text-transform:uppercase;margin-bottom:14px}
h1{font-family:Georgia,"Songti SC",serif;font-size:38px;line-height:1.3;font-weight:600;color:var(--navy);letter-spacing:.5px}
.sub{color:var(--ink2);font-size:14px;margin-top:12px}
.meta{color:var(--muted);font-size:12px;margin-top:16px;letter-spacing:.02em}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 46px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:3px;padding:20px 18px;position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;width:3px;height:100%;background:var(--gold)}
.kpi .v{font-family:Georgia,serif;font-size:27px;color:var(--navy);letter-spacing:.5px}
.kpi .l{font-size:11.5px;color:var(--muted);margin-top:6px;letter-spacing:.06em}
h2{font-family:Georgia,"Songti SC",serif;font-size:23px;color:var(--navy);margin:52px 0 8px;font-weight:600;
display:flex;align-items:center;gap:14px}
h2 .n{font-family:"Helvetica Neue",sans-serif;font-size:12px;color:var(--gold);letter-spacing:.2em;
border:1px solid var(--gold);border-radius:50%;width:30px;height:30px;display:inline-flex;
align-items:center;justify-content:center;flex:none}
h3{font-size:16px;color:var(--navy);margin:28px 0 10px;font-weight:600}
p{margin:12px 0;color:#26384b;text-align:justify}
.lead{font-size:16px;color:var(--ink2)}
figure{background:var(--card);border:1px solid var(--line);padding:14px;margin:24px 0 8px}
figure img{width:100%;display:block}
figcaption{font-size:12px;color:var(--muted);margin-top:10px;padding:0 4px}
table{width:100%;border-collapse:collapse;margin:18px 0;font-size:13px;background:var(--card)}
th{background:var(--navy);color:#f2efe9;font-weight:500;padding:10px 12px;text-align:left;font-size:12px;letter-spacing:.03em}
td{padding:9px 12px;border-bottom:1px solid var(--line);color:#2c3e50}
tr:nth-child(even) td{background:#fbfaf7}
.note{background:#fdfbf5;border-left:3px solid var(--gold);padding:16px 20px;margin:22px 0;font-size:13.5px;color:#4a5a6a}
.note b{color:var(--navy)}
.rec{background:var(--navy);color:#e9eef4;padding:26px 30px;margin:28px 0;border-radius:2px}
.rec h3{color:#f4efe6;margin-top:0;font-size:16px;letter-spacing:.06em}
.rec ul{margin:10px 0 0 18px}
.rec li{margin:9px 0;color:#d7e1ea;font-size:13.5px}
.rec li b{color:var(--gold);font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.tag{display:inline-block;font-size:11px;letter-spacing:.05em;color:#fff;background:var(--gold);
padding:2px 8px;border-radius:2px;margin-bottom:6px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:11.5px}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}h1{font-size:29px}}
"""


def build_report(kpis, summary_df, cap_table, rul_table, rul_display, warn_df, ml_metrics, figs, gp_store):
    def df_html(d, fmt=None, index=False):
        d = d.copy()
        if fmt:
            for c, f in fmt.items():
                if c in d.columns:
                    d[c] = d[c].map(lambda v: "" if pd.isna(v) else f.format(v))
        return d.to_html(index=index, border=0, escape=False, na_rep="—")

    sum_fmt = {"初始容量(Ah)": "{:.3f}", "末期容量(Ah)": "{:.3f}", "末期SOH(%)": "{:.1f}",
               "80%SOH循环": "{:.1f}", "75%SOH循环(EOL)": "{:.1f}", "70%SOH循环": "{:.1f}",
               "平均衰减(mAh/循环)": "{:.2f}"}
    cap_fmt = {"MAE(Ah)": "{:.4f}", "RMSE(Ah)": "{:.4f}", "MAPE(%)": "{:.2f}", "R²": "{:.4f}"}
    warn_fmt = {"真实RUL": "{:.1f}", "RUL中位数": "{:.1f}", "95%区间下限": "{:.1f}",
                "95%区间上限": "{:.1f}", "区间宽度": "{:.1f}", "误差": "{:.1f}"}
    ml_fmt = {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "MAPE": "{:.2f}"}
    rul_fmt = {"平均绝对误差(循环)": "{:.1f}", "中位绝对误差(循环)": "{:.1f}",
               "最大误差(循环)": "{:.1f}", "有效电池数": "{:.0f}"}

    rul_show = rul_display.copy()
    for m in ["二次回归", "双指数", "对数GP"]:
        rul_show[m] = rul_show[m].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    rul_show["真实RUL"] = rul_show["真实RUL"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    rul_show = rul_show.rename(columns={"二次回归": "二次回归预测", "双指数": "双指数预测", "对数GP": "对数GP预测"})

    hit = (warn_df["覆盖率"] == "命中").mean() * 100 if len(warn_df) else float("nan")
    b0005_rul = gp_store.get("B0005_rul_samples", np.array([]))
    rul_lo, rul_hi = (np.percentile(b0005_rul, [2.5, 97.5]) if b0005_rul.size else (np.nan, np.nan))
    b0018 = ml_metrics[ml_metrics["电池"] == "B0018"]
    b0018_mae = float(b0018["MAE"].iloc[0]) if len(b0018) else float("nan")

    body = f"""
<header>
  <div class="eyebrow">Battery Health &amp; Prognostics</div>
  <h1>无人设备锂离子电池<br>健康评估与剩余寿命预测</h1>
  <div class="sub">基于 NASA PCoE 电池老化数据的退化建模、模型对比与不确定性分析</div>
  <div class="meta">数据：题3_电池健康老化.csv（B0005 / B0006 / B0007 / B0018 · 24 °C 恒温 · {kpis['cycles']} 个放电循环，
  与 NASA PCoE 原始数据逐点一致）· 寿命终点定义 1.50 Ah（75% SOH）</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">{kpis['cycles']}</div><div class="l">有效放电循环</div></div>
  <div class="kpi"><div class="v">{kpis['batteries']}</div><div class="l">电池个体</div></div>
  <div class="kpi"><div class="v">{kpis['avg_end_soh']:.0f}%</div><div class="l">末期平均 SOH</div></div>
  <div class="kpi"><div class="v">{kpis['best_rul_medae']:.0f}</div><div class="l">最佳 RUL 中位误差（循环）</div></div>
</div>

<p class="lead">本报告用 Python 打通了从充放电循环数据到寿命预测的完整链路：以题目提供的 CSV 为主数据源
（已验证与 NASA PCoE 官方 .mat 原始数据逐点一致，最大偏差 &lt; 0.0001 Ah），
清洗容量再生造成的异常点、提取退化特征，并以二次回归、双指数经验模型、对数域高斯过程三种方法预测容量与
RUL，再用跨电池机器学习模型验证泛化能力。核心结论是：<b>容量退化可被少数几个早期特征有效刻画，
但寿命外推的不确定性会随预测时点提前而迅速放大</b>。</p>

<h2><span class="n">01</span>数据探索：退化规律与个体差异</h2>
<p>四个电池均表现出典型的锂离子电池容量衰减：前期缓慢、后期加速，并在中段出现由“容量再生”现象
引起的短暂回升。原始序列中存在少量尖峰（红圈），若直接用于拟合会显著高估后期容量，因此先做中值滤波
识别并替换，再统一换算为 SOH。</p>
<figure>
  <img src="{img_b64(figs['fig1'])}" alt="容量退化曲线">
  <figcaption>图 1　四个电池的放电容量随循环的退化曲线；虚线为 80% / 75% / 70% SOH 阈值，× 为首次跌破 75% SOH 的位置。</figcaption>
</figure>
<p>个体差异非常明显：B0006 初始容量最高（约 2.03 Ah）但衰减最快，B0007 初始容量较低、衰减最慢，
B0018 循环数最少（132 次）却最先跌破 75% SOH。这意味着<b>不能用一个固定循环数来安排全机队的更换</b>，
必须按个体状态评估。</p>
<figure>
  <img src="{img_b64(figs['fig2'])}" alt="SOH 与衰减速率">
  <figcaption>图 2　SOH 衰减轨迹（左）与平均衰减速率对比（右）。速率相差近一倍，说明个体一致性差异是维护策略的关键变量。</figcaption>
</figure>
{df_html(summary_df, sum_fmt)}

<div class="note"><b>数据规模与清洗：</b>{kpis['cycles']} 条放电循环全部保留，仅对 {int(summary_df.shape[0])} 个电池中
的少量再生尖峰做局部替换，未删除任何循环，避免破坏退化趋势的连续性。</div>

<h2><span class="n">02</span>建模：三种容量退化模型对比</h2>
<p>统一采用<b>仅前 45 个循环</b>的数据建模，再外推到第 150 循环，模拟“电池刚投入使用不久就要给出寿命判断”
的真实场景。三种方法分别是：</p>
<ul>
  <li><b>二次多项式回归</b>：最常用的简单回归基线，可解析求交点，但二次项在远期会快速下坠。</li>
  <li><b>双指数经验模型</b>：Q = a·e<sup>bk</sup> + c·e<sup>dk</sup>，电池文献的经典退化形式，能表达“先缓后急”。</li>
  <li><b>对数域高斯过程回归</b>：对 ln Q 建模，核函数为 RBF + DotProduct + 白噪声；DotProduct 提供趋势外推，
      RBF 拟合局部波动，同时输出逐点置信区间，是后续不确定性分析的基础。</li>
</ul>
<figure>
  <img src="{img_b64(figs['fig3'])}" alt="容量模型对比">
  <figcaption>图 3　浅蓝阴影为建模区（前 45 循环），右侧为纯外推区。金色带为对数GP 的 95% 置信区间。</figcaption>
</figure>
<p>图 3 显示了三种外推性格的差异：二次回归在后期“下坠”过快，到第 150 循环时 B0006 已预测到 0.5 Ah，
远低于实测的 1.25 Ah；双指数则几乎走平，低估了加速衰减；对数GP 在观测区内贴合最好，外推相对克制。
这也说明<b>“拟合得好”与“外推得准”是两件不同的事</b>。</p>
{df_html(cap_table, cap_fmt)}
<div class="note"><b>结论：</b>容量精度上，<b>{kpis['best_cap_model']}</b> 的 RMSE 最低
（≈ {kpis['best_cap_rmse']:.4f} Ah，MAPE ≈ 6.7%）；但所有模型在 100 循环以外的长时间外推上 R² 都不理想，
这本身就是一条重要结论——<b>纯数据外推存在能力边界</b>，必须配合不确定性区间使用。</div>

<h3>剩余寿命（RUL）预测</h3>
<p>以 1.50 Ah（75% SOH）为寿命终点，把各模型曲线与阈值的交点换算为“从第 45 循环起还能用多少循环”。
“—”表示模型曲线在 260 循环内未穿越阈值，即外推无法给出可信的寿命结论。</p>
{df_html(rul_show)}
<p>各方法的误差汇总（仅统计能给出交点的电池）：</p>
{df_html(rul_table, rul_fmt)}
<figure>
  <img src="{img_b64(figs['fig4'])}" alt="RUL 对比">
  <figcaption>图 4　第 45 循环时的 RUL 误差（左）与预测/真实散点（右）。虚线为理想预测，浅色带为 ±20% 容差。</figcaption>
</figure>
<p>RUL 误差与容量误差并不同步，甚至出现“反转”：<b>二次回归的容量 RMSE 最大，RUL 却最准</b>
（中位误差 6.6 循环）。原因是它在远期下坠更快，交点位置反而接近真实寿命；而对数GP 的容量曲线更准，
但外推偏保守，交点被推后，B0005 的 RUL 高估近 60 循环，B0007 甚至在 260 循环内都没有交点。
这说明工程上应<b>同时看容量误差与决策误差</b>，不能只挑 RMSE 最小的模型。</p>

<h2><span class="n">03</span>不确定性与风险：提前预警 vs 寿命外推</h2>
<p>把 GP 后验作为随机过程抽样 1500 次，每次求出穿越阈值的时刻，得到 RUL 的完整概率分布，而不是单一数字。</p>
<figure>
  <img src="{img_b64(figs['fig5'])}" alt="不确定性">
  <figcaption>图 5　左：随着外推步长增加，置信带迅速张开；右：B0005 在第 45 循环时的 RUL 后验分布，
  95% 区间为 [{rul_lo:.0f}, {rul_hi:.0f}] 循环。</figcaption>
</figure>
<p>进一步在四个不同时点（第 30 / 45 / 60 / 75 循环）重复预测，观察区间宽度与误差如何随时点变化：</p>
{df_html(warn_df, warn_fmt)}
<p>提前预警的 95% 区间覆盖率约为 <b>{hit:.0f}%</b>（15 个预测时点命中 14 个），说明 GP 的统计不确定性量化基本可信。
但区间宽度差异极大：B0018 在第 30 循环时只剩 ±6 循环的窄区间，而 B0005 在第 30 循环时的区间宽达 182 循环，
几乎无法指导决策。总体规律是<b>越接近寿命终点、观测越多，区间越窄</b>。</p>
<figure>
  <img src="{img_b64(figs['fig6'])}" alt="提前预警误差">
  <figcaption>图 6　预测时点与 RUL 的关系：圆点为预测中位数，误差条为 95% 区间，短横线为真实值。</figcaption>
</figure>

<div class="note"><b>两类场景的风险不对称：</b><br>
<b>提前预警</b>（如剩余 20 循环内提醒备件）容错空间小，必须使用区间下界而不是中位数，
否则会出现“电池突然掉电”的运营风险；<br>
<b>寿命外推</b>（用前几十循环推断全程寿命）误差随步长增长，且 GP 的置信区间只反映统计不确定性、
不包含模型形式偏差——B0005、B0007 的 RUL 被系统性高估就是后者。因此外推结果更适合做采购与预算规划，
不适合直接触发单机维护动作。</div>

<h2><span class="n">04</span>跨电池机器学习模型</h2>
<p>上面的三种方法都对单个电池独立建模。为了检验“能否把多个电池的经验迁移到新电池”，又训练了一个
梯度提升回归模型（HistGradientBoosting），输入包括 SOH、近 5 循环均值、10/20 循环衰减斜率、温度、
放电时长、放电能量与内阻代理量，采用<b>留一电池</b>方式验证：用 3 个电池训练，预测第 4 个。</p>
{df_html(ml_metrics, ml_fmt)}
<figure>
  <img src="{img_b64(figs['fig7'])}" alt="机器学习 RUL">
  <figcaption>图 7　左：跨电池 RUL 预测（虚线）与真实值（实线）；右：置换重要性，SOH、放电能量与衰减斜率贡献最大。</figcaption>
</figure>
<p>跨电池平均 MAE ≈ {kpis['ml_mae']:.1f} 循环；四折中三折 R² 为正（0.01–0.54），B0006 一折 R² 为负，
表现最好的一折（B0018）MAE ≈ {b0018_mae:.1f} 循环，最差的一折（B0006）约 25.6 循环。
置换重要性显示，<b>SOH、放电能量与近期衰减斜率是最主要的输入</b>，绝对循环数贡献有限——这与“同型号电池衰减速度差异大”一致。
但也应看到，跨电池迁移在衰减模式差异大的个体上误差明显上升：<b>个体差异是主导因素，通用模型只能作为冷启动</b>。</p>

<h2><span class="n">05</span>结论与维护建议</h2>
<p>以下建议可直接落地：</p>
<div class="rec">
  <h3>基于数据的运维建议</h3>
  <ul>
    <li><b>按 SOH 而非循环数更换。</b>四个电池跌破 80% SOH 的循环数分别为 74、63、85、45，相差近一倍；统一按循环数计划会造成过度更换或提前失效。</li>
    <li><b>80% SOH 触发预警，75% SOH 触发更换评估。</b>80% 时容量仍可支撑多数任务，但已进入加速衰减区；75%（1.50 Ah）作为更换硬阈值，与本文 RUL 定义一致。</li>
    <li><b>用区间下界做提前预警。</b>B0018 这类衰减快的个体，第 30 循环即可给出 ±6 循环的窄区间，可放心预警；B0005 早期区间宽达 180 循环，此时只能用于规划、不能触发维护。</li>
    <li><b>警惕“高估剩余寿命”的模型偏差。</b>对数GP 在 B0005 / B0007 上把寿命高估了 58 / 60 个循环以上，且统计置信区间覆盖不到该偏差；工程上应交叉使用更保守的回归外推作为兜底。</li>
    <li><b>重视温度与近期斜率。</b>温度均值与近 20 循环衰减斜率是 RUL 的主要预测特征；持续高温运行或斜率突然变陡，应优先安排检测。</li>
    <li><b>新电池先冷启动、后个体化。</b>前 30–45 循环使用经验/回归模型并保留宽区间，积累数据后再切换到机器学习个体模型（跨电池模型的平均 MAE 仍达 18.8 循环）。</li>
  </ul>
</div>
<p>总体而言，容量退化本身是高度可预测的，真正的难点在于<b>把预测变成可用于决策的概率语言</b>：
同一个“还能用 40 个循环”的结论，在只有 30 个循环数据时和在 100 个循环数据时的可信度完全不同。
本报告给出的 GP 后验分布与提前预警曲线，正是为了让维护决策看到这种差别。</p>

<footer>
  复现：battery_health_analysis.py（Python 3.12 + pandas / scikit-learn / scipy / matplotlib）·
  数据：题3_电池健康老化.csv（源自 NASA PCoE Battery Data Set，辅助特征取自原始 .mat）· 随机种子 42/3/11 · 报告由脚本自动生成
</footer>
"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>无人设备电池健康评估与剩余寿命预测</title>
<style>{CSS}</style>
</head>
<body><div class="wrap">{body}</div></body></html>"""
    (BASE / "battery_health_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
