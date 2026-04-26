import matplotlib.pyplot as plt
import pandas as pd
from typing import List

from config import CountryConfig
from equations import LAMBDA0

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, countries: List[CountryConfig],
                 save_path: str = None):
    """
    Plot key variables over 260 weeks.
    """
    names = [c.name for c in countries]
    weeks = df["week"].values

    colors = ["#185FA5", "#c0392b", "#2a7a2a", "#d68910", "#7d3c98"]
    color_map = {n: colors[i % len(colors)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("CARMA AI Arms Race Simulation\n(Welfare corrected: (1−Gini)×GDP per capita)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")

    # 1. AI Potency
    ax = axes[0, 0]
    for n in names:
        ax.plot(weeks, df[f"{n}_p"], label=n, color=color_map[n], linewidth=1.8)
    ax.set_title("AI Potency (p)", fontsize=11)
    ax.set_ylabel("Potency (1–100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Domestic Regulation
    ax = axes[0, 1]
    for n in names:
        ax.plot(weeks, df[f"{n}_reg_dom"], label=n, color=color_map[n], linewidth=1.8)
    ax.plot(weeks, df["reg_int"], label="reg_int (global)",
            color="black", linewidth=1.5, linestyle="--")
    ax.set_title("Regulation (domestic + international)", fontsize=11)
    ax.set_ylabel("Regulation index [0,1]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Gini Inequality
    ax = axes[0, 2]
    for n in names:
        ax.plot(weeks, df[f"{n}_gini"], label=n, color=color_map[n], linewidth=1.8)
    ax.set_title("Gini Inequality", fontsize=11)
    ax.set_ylabel("Gini coefficient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Welfare — CORRECTED: (1 - Gini) * GDP per capita
    ax = axes[1, 0]
    for n in names:
        ax.plot(weeks, df[f"{n}_welfare"] / 1000, label=n,
                color=color_map[n], linewidth=1.8)
    ax.set_title("Welfare = (1−Gini) × GDP/capita  ← corrected", fontsize=11,
                 color="#185FA5")
    ax.set_ylabel("Welfare index (thousands USD)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5. Biorisk — attacks per week
    ax = axes[1, 1]
    ax.plot(weeks, df["biorisk_attacks_per_week"], color="#c0392b", linewidth=1.8)
    ax.axhline(LAMBDA0, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline λ₀ = {LAMBDA0:.4f}")
    ax.set_title("Biorisk: expected attacks/week", fontsize=11)
    ax.set_ylabel("λ (attacks/week)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 6. Biorisk — expected deaths
    ax = axes[1, 2]
    ax.plot(weeks, df["biorisk_expected_deaths"], color="#922b21", linewidth=1.8)
    ax.set_title("Biorisk: expected deaths/week", fontsize=11)
    ax.set_ylabel("Expected deaths")
    ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Week")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved: {save_path}")
    plt.close()