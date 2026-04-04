"""
VISUALIZATION
Plots the full pipeline: raw signals, outlier detection, reconciliation, and KPIs.
"""
import matplotlib.pyplot as plt


def plot_outliers(df):
    """Figure 1: Raw SCADA signals with detected outliers highlighted."""
    print("-> Generating Figure 1: Raw Data & Outlier Analysis (IQR)...")

    fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "SCADA Anomaly Detection — Ethanol Distillation Column (IQR Method)",
        fontsize=14, fontweight="bold",
    )

    # --- Mass flow streams ---
    mass_cols = ["F_feed_raw", "F_top_raw", "F_bottom_raw"]
    mass_titles = ["Feed Flow", "Top (Distillate) Flow", "Bottom Flow"]
    mass_colors = ["gray", "tab:blue", "tab:orange"]

    for i, (col, title, color) in enumerate(
        zip(mass_cols, mass_titles, mass_colors)
    ):
        axs[i, 0].plot(
            df["Timestamp"], df[col], color=color, alpha=0.7, label="Raw Signal"
        )
        outliers = df[df[f"{col}_outlier"] == True]
        axs[i, 0].scatter(
            outliers["Timestamp"], outliers[col],
            color="red", s=50, zorder=5, label="Outlier (IQR)",
        )
        axs[i, 0].set_title(title)
        axs[i, 0].set_ylabel("Flow (kg/h)")
        axs[i, 0].legend(loc="upper right")
        axs[i, 0].grid(True, linestyle="--", alpha=0.5)

    # --- Energy streams ---
    energy_cols = ["Q_cond_raw", "Q_reb_raw"]
    energy_titles = ["Condenser Duty (R_cond)", "Reboiler Duty (Q_reb)"]
    energy_colors = ["tab:cyan", "tab:red"]

    for i, (col, title, color) in enumerate(
        zip(energy_cols, energy_titles, energy_colors)
    ):
        axs[i, 1].plot(
            df["Timestamp"], df[col], color=color, alpha=0.7, label="Raw Signal"
        )
        outliers = df[df[f"{col}_outlier"] == True]
        axs[i, 1].scatter(
            outliers["Timestamp"], outliers[col],
            color="red", s=50, zorder=5, label="Outlier (IQR)",
        )
        axs[i, 1].set_title(title)
        axs[i, 1].set_ylabel("Duty (kW)")
        axs[i, 1].legend(loc="upper right")
        axs[i, 1].grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplot (row 3, col 2)
    axs[2, 1].axis("off")

    axs[2, 0].set_xlabel("Time")
    axs[1, 1].set_xlabel("Time")
    plt.tight_layout()


def plot_dashboard(df):
    """Figure 2: Reconciliation results and KPI tracking."""
    print("-> Generating Figure 2: Reconciliation & KPI Dashboard...")

    fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        "Digital Twin Results — Ethanol Column SCOL-1",
        fontsize=14, fontweight="bold",
    )

    # --- Panel 1: Feed signal treatment ---
    axs[0].plot(df["Timestamp"], df["F_feed_raw"], color="lightgray", label="Raw")
    axs[0].plot(
        df["Timestamp"], df["F_feed_filtered"],
        color="blue", alpha=0.6, label="Filtered",
    )
    axs[0].plot(
        df["Timestamp"], df["F_feed_rec"],
        color="red", linestyle="--", label="Reconciled",
    )
    axs[0].set_title("Signal Treatment: Feed Flow ($F_{feed}$)")
    axs[0].set_ylabel("Flow (kg/h)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Panel 2: Mass balance closure ---
    axs[1].plot(
        df["Timestamp"], df["Error_Mass_Before_%"],
        color="orange", label="Error before reconciliation",
    )
    axs[1].axhline(y=0, color="green", linestyle="--", label="Reconciled balance (0%)")
    axs[1].axhline(
        y=config_max_error(), color="red", linestyle=":",
        label=f"Tolerance ({config_max_error()}%)",
    )
    axs[1].set_title("Mass Conservation — Balance Closure")
    axs[1].set_ylabel("Closure Error (%)")
    axs[1].legend()
    axs[1].grid(True)

    # --- Panel 3: Separation adherence KPI ---
    axs[2].plot(
        df["Timestamp"], df["KPI_Separation_%"],
        color="purple", label="Separation Adherence",
    )
    axs[2].axhline(y=100, color="gray", linestyle=":")
    axs[2].set_title("KPI: Adherence to Target Distillate Split (35%)")
    axs[2].set_ylabel("Adherence (%)")
    axs[2].legend()
    axs[2].grid(True)

    # --- Panel 4: Energy efficiency KPI ---
    axs[3].plot(
        df["Timestamp"], df["KPI_Energy_%"],
        color="teal", label="Energy Efficiency",
    )
    axs[3].axhline(y=100, color="gray", linestyle=":")
    axs[3].set_title("KPI: Energy Ratio Adherence (Q_cond / Q_reb)")
    axs[3].set_ylabel("Efficiency (%)")
    axs[3].set_xlabel("Time")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()


def show_plots():
    """Render all figure windows."""
    plt.show()


def config_max_error():
    """Import lazily to avoid circular imports."""
    import config
    return config.MAX_MASS_BALANCE_ERROR