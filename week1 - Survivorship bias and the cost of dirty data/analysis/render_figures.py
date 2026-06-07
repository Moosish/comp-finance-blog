"""
render_figures.py
Reads data/week1_chart_data.json and writes 4 PNG figures to figures/.
Run:  python render_figures.py
"""
import json
import sys
import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))   # .../computationalfinance/
from chart_style import (
    apply, save_fig,
    C_PRIMARY, C_BLUE, C_AMBER, C_RED,
    TEXT_HIGH, TEXT_MED, TEXT_LOW, BG_AXES, BORDER,
)

apply()

FIGS = _HERE / "figures"
FIGS.mkdir(exist_ok=True)

with open(_HERE / "data" / "week1_chart_data.json") as f:
    D = json.load(f)


def _dates(section):
    return [datetime.strptime(s, "%Y-%m-%d") for s in D[section]["dates"]]


def _arr(section, series):
    return np.array(
        [x if x is not None else np.nan for x in D[section][series]], dtype=float
    )


def _event_lines(ax, y_text):
    for key, label in D["events"].items():
        dt = datetime.strptime(key + "-01", "%Y-%m-%d")
        ax.axvline(dt, color=TEXT_LOW, lw=0.75, ls=":", alpha=0.5)
        ax.text(dt, y_text, label, color=TEXT_LOW, fontsize=7.5,
                rotation=90, va="bottom", ha="right")


# ===========================================================================
# Figure 1 — Cumulative returns (log scale)
# ===========================================================================
cum_dates  = _dates("cumulative_returns")
biased_cum = _arr("cumulative_returns", "biased")
ref_cum    = _arr("cumulative_returns", "reference")

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(cum_dates, biased_cum, color=C_PRIMARY, lw=2,
        label="Biased (survivors-only)")
ax.plot(cum_dates, ref_cum,    color=C_BLUE,    lw=2, alpha=0.85,
        label="Reference (simulated PIT)")
ax.fill_between(cum_dates, ref_cum, biased_cum,
                where=biased_cum >= ref_cum,
                color=C_AMBER, alpha=0.12, label="Survivorship premium")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), ha="center")

ylo = float(np.nanmin([biased_cum.min(), ref_cum.min()])) * 0.93
yhi = float(biased_cum.max()) * 1.10
ax.set_ylim(ylo, yhi)
_event_lines(ax, y_text=ylo * 1.04)

ax.set_title("Cumulative Returns: Biased vs Reference Portfolio",
             loc="left", color=TEXT_HIGH, fontsize=12, fontweight="bold")
ax.text(0, 1.025,
        "Log scale  ·  Equal-weight monthly  ·  2010–2026  ·  SIMULATED",
        transform=ax.transAxes, color=TEXT_LOW, fontsize=8)
ax.set_ylabel("Portfolio value (×)", color=TEXT_MED)
ax.legend(loc="upper left")

save_fig(fig, FIGS / "fig1_cumulative_returns.png")
plt.close(fig)
print("done: fig1_cumulative_returns.png")


# ===========================================================================
# Figure 2 — Metrics comparison (Sharpe / Calmar / Max Drawdown)
# ===========================================================================
m_b = D["metrics"]["biased"]
m_r = D["metrics"]["reference"]

PANELS = [
    ("Sharpe Ratio",  m_b["Sharpe Ratio"],    m_r["Sharpe Ratio"],    False),
    ("Calmar Ratio",  m_b["Calmar Ratio"],     m_r["Calmar Ratio"],    False),
    ("Max Drawdown",  m_b["Max Drawdown"],     m_r["Max Drawdown"],    True),   # negative: closer to 0 = better
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.38)

for ax_i, (title, val_b, val_r, is_pct) in zip(axes, PANELS):
    bars_b = ax_i.bar([0], [val_b], width=0.5, color=C_PRIMARY, alpha=0.90)
    bars_r = ax_i.bar([1], [val_r], width=0.5, color=C_BLUE,    alpha=0.90)

    for bar, val in [(bars_b[0], val_b), (bars_r[0], val_r)]:
        label = f"{val:.1%}" if is_pct else f"{val:.3f}"
        if val >= 0:
            y_pos = bar.get_height() + bar.get_height() * 0.025
            va = "bottom"
        else:
            # For negative bars: bar.get_y() is the bar bottom (the negative value)
            y_pos = bar.get_y() - bar.get_height() * 0.04
            va = "top"
        ax_i.text(bar.get_x() + bar.get_width() / 2, y_pos,
                  label, ha="center", va=va,
                  color=TEXT_HIGH, fontsize=9.5, fontweight="bold")

    if title == "Sharpe Ratio":
        bias_pct = (val_b - val_r) / val_r * 100
        ax_i.text(0.5, 0.97, f"+{bias_pct:.1f}% inflation",
                  transform=ax_i.transAxes, ha="center", va="top",
                  color=C_AMBER, fontsize=9, fontweight="bold")

    if min(val_b, val_r) < 0:
        ax_i.set_ylim(min(val_b, val_r) * 1.30, max(val_b, val_r, 0) + abs(min(val_b, val_r)) * 0.15)
        ax_i.axhline(0, color=BORDER, lw=0.8)
        ax_i.text(0.98, 0.97, "closer to 0 = better",
                  transform=ax_i.transAxes, ha="right", va="top",
                  color=TEXT_LOW, fontsize=8)
    else:
        ax_i.set_ylim(0, max(val_b, val_r) * 1.30)

    ax_i.set_xlim(-0.5, 1.5)
    ax_i.set_xticks([0, 1])
    ax_i.set_xticklabels(["Biased", "Reference"], color=TEXT_MED, fontsize=9)
    ax_i.set_title(title, loc="center",
                   color=TEXT_HIGH, fontsize=11, fontweight="bold")
    if is_pct:
        ax_i.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

fig.suptitle("Portfolio Metrics: Biased vs Simulated Point-in-Time",
             y=1.03, color=TEXT_HIGH, fontsize=12, fontweight="bold")
fig.text(0.5, -0.02, "SIMULATED reference — see methodology",
         ha="center", color=TEXT_LOW, fontsize=8)

save_fig(fig, FIGS / "fig2_sharpe_comparison.png")
plt.close(fig)
print("done: fig2_sharpe_comparison.png")


# ===========================================================================
# Figure 3 — Rolling 3-year Sharpe ratio
# ===========================================================================
rs_dates  = _dates("rolling_sharpe")
biased_rs = _arr("rolling_sharpe", "biased")
ref_rs    = _arr("rolling_sharpe", "reference")

valid = ~np.isnan(biased_rs) & ~np.isnan(ref_rs)
rs_d  = [rs_dates[i] for i in range(len(rs_dates)) if valid[i]]
rs_b  = biased_rs[valid]
rs_r  = ref_rs[valid]

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(rs_d, rs_b, color=C_PRIMARY, lw=2,         label="Biased (survivors-only)")
ax.plot(rs_d, rs_r, color=C_BLUE,    lw=2, alpha=0.85, label="Reference (simulated PIT)")
ax.fill_between(rs_d, rs_r, rs_b,
                where=rs_b >= rs_r,
                color=C_AMBER, alpha=0.12, label="Survivorship premium")
ax.axhline(0, color=BORDER, lw=0.8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), ha="center")

ylo = float(np.nanmin([rs_b, rs_r])) - 0.15
yhi = float(np.nanmax([rs_b, rs_r])) + 0.15
ax.set_ylim(ylo, yhi)
_event_lines(ax, y_text=ylo + (yhi - ylo) * 0.02)

ax.set_title("Rolling 3-Year Sharpe Ratio",
             loc="left", color=TEXT_HIGH, fontsize=12, fontweight="bold")
ax.text(0, 1.025, "36-month rolling window  ·  Monthly data  ·  SIMULATED",
        transform=ax.transAxes, color=TEXT_LOW, fontsize=8)
ax.set_ylabel("Sharpe ratio", color=TEXT_MED)
ax.legend(loc="upper left")

save_fig(fig, FIGS / "fig3_rolling_bias.png")
plt.close(fig)
print("done: fig3_rolling_bias.png")


# ===========================================================================
# Figure 4 — Annual Sharpe gap decomposition
# ===========================================================================
years   = D["decomposition"]["years"]
ret_inf = np.array(D["decomposition"]["return_inflation"])
vol_sup = np.array(D["decomposition"]["vol_suppression"])
total   = np.array(D["decomposition"]["total_gap"])

x_pos = np.arange(len(years))
bw    = 0.35

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.bar(x_pos - bw / 2, ret_inf, width=bw, color=C_PRIMARY, alpha=0.88,
       label="Return inflation")
ax.bar(x_pos + bw / 2, vol_sup, width=bw, color=C_BLUE,    alpha=0.88,
       label="Volatility suppression")
ax.plot(x_pos, total, color=C_AMBER, lw=2, marker="o",
        markersize=4.5, markerfacecolor=C_AMBER, markeredgewidth=0,
        label="Total Sharpe gap", zorder=5)
ax.axhline(0, color=BORDER, lw=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(years, fontsize=8.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.2f}"))

ax.set_title("Annual Sharpe Gap: Return Inflation vs Volatility Suppression",
             loc="left", color=TEXT_HIGH, fontsize=12, fontweight="bold")
ax.text(0, 1.025, "Positive = biased Sharpe > reference Sharpe  ·  SIMULATED",
        transform=ax.transAxes, color=TEXT_LOW, fontsize=8)
ax.set_ylabel("Sharpe gap contribution", color=TEXT_MED)
ax.legend(loc="upper right", ncol=3)

save_fig(fig, FIGS / "fig4_return_decomposition.png")
plt.close(fig)
print("done: fig4_return_decomposition.png")

print(f"\nAll figures saved: {FIGS}")
