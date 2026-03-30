"""
Digital Twin package for distillation column process monitoring.

This package implements a complete digital twin pipeline:
- DWSIM integration (mock mode) for distillation column simulation
- Excel sensor data ingestion
- Data treatment: IQR outlier detection/removal
- Signal filtering: moving average or low-pass Butterworth filter
- Data reconciliation: Weighted Least Squares (WLS)
- KPIs and adherence indicators

References:
-----------
- Narasimhan, S., & Jordache, C. (1999). Data Reconciliation & Gross Error Detection.
  Gulf Professional Publishing.
- Mah, R. S. H. (1990). Chemical Process Structures and Information Flows.
  Butterworth-Heinemann.
"""
