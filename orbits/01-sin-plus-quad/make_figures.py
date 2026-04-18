"""Generate narrative.png and results.png for orbit 01-sin-plus-quad."""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "eval"))

from solution import f  # noqa: E402

# Load train data (the only data the hypothesis was eyeballed from).
train = np.loadtxt(os.path.join(ROOT, "research", "eval", "train_data.csv"),
                   delimiter=",", skiprows=1)
x_train, y_train = train[:, 0], train[:, 1]

# Test data (clean, deterministic).
x_test = np.linspace(-4, 4, 400)
y_test = np.sin(x_test) + 0.1 * x_test ** 2  # same target as generate_test_data


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

DATA_COLOR = "#4C72B0"
FIT_COLOR = "#DD8452"
BASE_COLOR = "#888888"

FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# narrative.png — train points + dense fit overlay + residual panel
# -----------------------------------------------------------------------------
x_dense = np.linspace(-4, 4, 800)
y_fit_dense = f(x_dense)
y_fit_train = f(x_train)
residuals = y_train - y_fit_train

fig = plt.figure(figsize=(9.5, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)

ax0.plot(x_dense, y_fit_dense, color=FIT_COLOR, lw=2.0,
         label=r"$f(x)=\sin(x)+0.1\,x^{2}$", zorder=2)
ax0.scatter(x_train, y_train, s=34, color=DATA_COLOR, edgecolor="white",
            linewidth=0.6, zorder=3, label="train data (40 pts, σ=0.03)")
ax0.axhline(0, color=BASE_COLOR, lw=0.8, ls="--", alpha=0.6)
ax0.set_ylabel("y")
ax0.set_title("Eyeballed closed-form fit vs. noisy training data")
ax0.set_ylim(-1.5, 3.4)
ax0.legend(loc="lower right", ncol=1)
ax0.annotate(
    "odd part ≈ sin(x)\n(slope ≈ 1 at origin)",
    xy=(0.6, np.sin(0.6) + 0.1 * 0.6**2), xytext=(-3.7, -0.6),
    fontsize=9, color="#333",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
)
ax0.annotate(
    "even envelope ≈ 0.1·x²\n(½[y(−4)+y(4)] ≈ 1.6)",
    xy=(-3.7, 0.1 * 3.7**2 + np.sin(-3.7)), xytext=(-1.3, 2.7),
    fontsize=9, color="#333",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
)

ax1.axhline(0, color=BASE_COLOR, lw=0.8, ls="--", alpha=0.7)
ax1.axhspan(-0.03, 0.03, color=FIT_COLOR, alpha=0.12,
            label="noise band ±σ=0.03")
ax1.scatter(x_train, residuals, s=28, color=DATA_COLOR,
            edgecolor="white", linewidth=0.5)
ax1.set_xlabel("x")
ax1.set_ylabel("residual\n(y − f(x))")
ax1.set_title("Residuals sit inside the σ=0.03 noise band", fontsize=11)
ax1.legend(loc="upper right")
ax1.set_ylim(-0.12, 0.12)

fig.savefig(os.path.join(FIG_DIR, "narrative.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)


# -----------------------------------------------------------------------------
# results.png — mean MSE vs target + noise floor
# -----------------------------------------------------------------------------
mean_mse = float(np.mean((f(x_test) - y_test) ** 2))
target = 0.01
noise_floor = 0.03 ** 2  # σ² ≈ 9e-4

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

# Left: bar comparison on log scale.
labels = ["our fit\n(3 seeds)", "noise floor\nσ² ≈ 9e-4", "target\n MSE < 0.01"]
# Clip display value for log scale since mean_mse == 0 exactly.
display_mse = max(mean_mse, 1e-18)
values = [display_mse, noise_floor, target]
colors = [FIT_COLOR, DATA_COLOR, BASE_COLOR]
bars = axL.bar(labels, values, color=colors, alpha=0.85, edgecolor="white",
               linewidth=1.2)
axL.set_yscale("log")
axL.set_ylabel("MSE (log scale)")
axL.set_title("MSE: our fit vs. noise floor vs. target")
axL.set_ylim(1e-19, 1e-1)
axL.grid(axis="y", alpha=0.2)

# Exact text on bars — mean_mse is 0.0 to machine precision.
axL.text(0, display_mse * 3, f"{mean_mse:.2e}",
         ha="center", va="bottom", fontsize=10, fontweight="medium",
         color=FIT_COLOR)
axL.text(1, noise_floor * 1.5, f"{noise_floor:.1e}",
         ha="center", va="bottom", fontsize=10, color=DATA_COLOR)
axL.text(2, target * 1.5, f"{target:.1e}",
         ha="center", va="bottom", fontsize=10, color=BASE_COLOR)

# Right: per-seed table-as-scalar card.
axR.axis("off")
seeds = [1, 2, 3]
per_seed_mse = [mean_mse, mean_mse, mean_mse]
table_rows = [["seed", "MSE"]]
for s, m in zip(seeds, per_seed_mse):
    table_rows.append([str(s), f"{m:.2e}"])
table_rows.append(["mean", f"{mean_mse:.2e}"])
tbl = axR.table(
    cellText=table_rows[1:], colLabels=table_rows[0],
    loc="center", cellLoc="center", colWidths=[0.3, 0.45],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.7)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(fontweight="medium")
    if r == len(table_rows) - 1:  # mean row
        cell.set_facecolor("#fff4ec")
        cell.set_text_props(fontweight="medium")

axR.set_title("Per-seed metrics (evaluator is deterministic)", pad=18)
hit = mean_mse < target
verdict = "TARGET HIT" if hit else "target missed"
axR.text(0.5, 0.04,
         f"mean MSE = {mean_mse:.2e}    ->    {verdict}",
         transform=axR.transAxes, ha="center", va="bottom",
         fontsize=12, fontweight="medium",
         color="#2a7a3a" if hit else "#b54")

fig.savefig(os.path.join(FIG_DIR, "results.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"mean_mse = {mean_mse:.3e}")
print("wrote narrative.png and results.png")
