"""
MAIN SCRIPT
Orchestrates the full Digital Twin pipeline for ethanol distillation:
  1. Generate SCADA data (from DWSIM or synthetic)
  2. Clean data (IQR outlier removal + moving-average filter)
  3. Reconcile mass balance via constrained optimization
  4. Compute KPIs (mass closure, separation adherence, energy efficiency)
  5. Visualize results
"""
import data_processor
import reconciliation_engine
import dashboard
import config


def main():
    print("=" * 60)
    print("  ETHANOL DISTILLATION — DIGITAL TWIN PIPELINE")
    print("=" * 60)

    # Step 1: Generate raw data (live DWSIM or synthetic)
    if config.USE_DWSIM_LIVE:
        print("🔗 Using live DWSIM simulation...")
        import dwsim_data_generator
        is_valid, message = dwsim_data_generator.validate_dwsim_installation()
        if not is_valid:
            print(f"❌ DWSIM validation failed: {message}")
            print("⚠️  Falling back to synthetic data generation...")
            df_raw = data_processor.generate_raw_data()
        else:
            print(f"✅ DWSIM validation successful: {message}")
            df_raw = dwsim_data_generator.generate_dwsim_data(config.N_POINTS)
    else:
        print("📊 Using synthetic SCADA data...")
        df_raw = data_processor.generate_raw_data()

    # Step 2: Clean data
    df_filtered = data_processor.clean_and_filter(df_raw)

    # Step 3 & 4: Reconcile balances and compute KPIs
    df_final = reconciliation_engine.reconcile_balances(df_filtered)

    # Step 5: Visualize
    dashboard.plot_outliers(df_final)
    dashboard.plot_dashboard(df_final)

    print("=" * 60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("  Close the plot windows to end.")
    print("=" * 60)

    dashboard.show_plots()


if __name__ == "__main__":
    main()