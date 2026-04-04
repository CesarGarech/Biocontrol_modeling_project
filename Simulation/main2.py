"""
MAIN SCRIPT
Orchestrates the full Digital Twin pipeline for ethanol distillation:
  1. Generate synthetic SCADA data with noise and outliers
  2. Clean data (IQR outlier removal + moving-average filter)
  3. Reconcile mass balance via constrained optimization
  4. Compute KPIs (mass closure, separation adherence, energy efficiency)
  5. Visualize results
"""
import data_processor
import reconciliation_engine
import dashboard


def main():
    print("=" * 60)
    print("  ETHANOL DISTILLATION — DIGITAL TWIN PIPELINE")
    print("=" * 60)

    # Step 1 & 2: Generate raw data, detect outliers, filter
    df_raw = data_processor.generate_raw_data()
    df_filtered = data_processor.clean_and_filter(df_raw)

    # Step 3 & 4: Reconcile balances and compute KPIs
    df_final = reconciliation_engine.reconcile_balances(df_filtered)

    # Step 5: Visualize
    dashboard.plot_outliers(df_final)
    dashboard.plot_dashboard(df_final)

    print("=" * 60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("  Close the plot windows to end execution.")
    print("=" * 60)

    dashboard.show_plots()


if __name__ == "__main__":
    main()