# Modeling and control applied to bioprocess
> An interactive Streamlit application for teaching modeling, simulation, analysis, and control of bioprocesses.

## 📖 Overview
This repository contains the source code for Biocontrol_modeling_project, developed by César Augusto García Echeverry at the Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ.

The purpose of this project is to assist in teaching classical and advanced modeling and control techniques within the fields of bioprocessing and biotechnology. It provides interactive tools built using robust programming methods, serving as a resource for education, research, development, and collaboration.

## 🚀 Features
- **Interactive Bioprocess Simulation:** Simulate various bioreactor operation modes (Batch, Fed-Batch, Continuous) using ODE models solved with `scipy`.
- **Specific Fermentation Modeling:** Includes a dedicated model for alcoholic fermentation simulating batch and fed-batch phases with complex yeast kinetics.
- **Kinetic Model Library:** Centralized definitions for various specific growth rate ($\mu$) models (Monod, Sigmoidal, Substrate/Product/Oxygen Inhibition) in `Utils/kinetics.py`, including CasADi-compatible versions.
- **Parameter Estimation:** Fit kinetic model parameters to experimental data (loaded from Excel files) using optimization algorithms (`scipy.optimize`) and perform statistical analysis (R², RMSE, confidence intervals).
- **Sensitivity Analysis:** Evaluate the impact of kinetic parameter variations on simulation outcomes.
- **State Estimation:** Implement an Extended Kalman Filter (EKF) using CasADi to estimate unmeasurable states (Biomass, Substrate, Product) and parameters from noisy simulated measurements.
- **Regulatory Process Control:** Simulate basic PID control loops for Temperature, pH (split-range), Dissolved Oxygen (via agitation), and On/Off substrate feeding.
- **Advanced Process Control:** Implement Real-Time Optimization (RTO) and Nonlinear Model Predictive Control (NMPC) using CasADi and IPOPT to find optimal operating profiles (e.g., feed rate) subject to constraints.
- **Interactive User Interface:** Built with Streamlit (`main.py`) for easy navigation and visualization using Matplotlib.

## 📦 Installation

### Option 1: Quick Start (Windows)
Just run the provided batch script:
```bash
run_dashboard.bat
```
This will automatically create a virtual environment, install dependencies, and launch the dashboard.

### Option 2: Manual Installation
Ensure you have Python 3.8 or higher installed. Then:

1. Clone the repository:
```bash
git clone https://github.com/CesarGarech/Biocontrol_modeling_project.git
cd Biocontrol_modeling_project
```

2. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
# Option A: Install in development mode
pip install -e .

# Option B: Install just the requirements
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main.py
```


## 📂 Repository Structure

```text
Biocontrol_modeling_project/
├── Body/                   # Core modules: modeling, analysis, estimation, control
│   ├── __init__.py
│   ├── modeling/           # Bioreactor simulation modes (batch, fed-batch, etc.)
│   │   ├── __init__.py
│   │   ├── lote.py        # Batch reactor
│   │   ├── lote_alimentado.py  # Fed-batch reactor
│   │   ├── continuo.py    # Continuous reactor (chemostat)
│   │   └── ferm_alcohol.py     # Alcoholic fermentation
│   ├── analysis.py         # Sensitivity analysis code
│   ├── parameter_estimation/  # Parameter estimation modules
│   │   ├── __init__.py
│   │   ├── ajuste_parametros_lote.py       # Batch parameter fitting
│   │   ├── ajuste_parametros_lote_alim.py  # Fed-batch parameter fitting
│   │   └── ajuste_parametros_ferm.py       # Fermentation parameter fitting
│   ├── estimation/         # State estimation (EKF, ANN)
│   │   ├── __init__.py
│   │   ├── ekf.py         # Extended Kalman Filter
│   │   └── ann.py         # Artificial Neural Network
│   ├── control/            # Process control strategies
│   │   ├── __init__.py
│   │   ├── regulatorio/   # Regulatory (PID) control
│   │   │   ├── __init__.py
│   │   │   ├── reg_temp.py, reg_ph.py, reg_oxigeno.py, etc.
│   │   └── avanzado/      # Advanced control (RTO, NMPC)
│   │       ├── __init__.py
│   │       ├── rto.py, rto_ferm.py, nmpc.py
│   └── home.py            # Home page content
├── Utils/                  # Utility functions
│   ├── __init__.py
│   └── kinetics.py         # Kinetic model definitions (Monod, etc.)
├── Data/                   # Example experimental datasets (.xlsx)
├── Examples/               # Standalone examples (EKF, RTO, NMPC CasADi scripts)
├── test_data/              # Test data files
├── Images/                 # Images for documentation and UI
├── Output/                 # Output files from simulations
├── setup.py                # Package setup configuration
├── main.py                 # Main Streamlit application entry point
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies list
└── run_dashboard.bat       # Windows batch script to run the dashboard
```

## ✏️ Authors & Contributors

This project was primarily developed by César Augusto García Echeverry at the Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ.

-**César Augusto García Echeverry - Lead Developer - [cesar.garech@gmail.com]**

-**Andres Mateo Franco Reyes - Developer - [anfrancor@unal.edu.co]**

-**Juan Pablo Velez Orjuela - Developer - [jvelezor@unal.edu.co]**

-**Carolina del Mar Gámez Herazo - Developer - [carolina.gamez@eia.edu.co]**


We welcome contributions! Please refer to the contribution guidelines (if available) or open an issue/pull request.


## 🔬 References & Publications
If you use this work in your research, please cite the following publications (if applicable):

Author(s). "Title of Paper." Journal/Conference, Year. DOI/Link

GitHub Repository: https://github.com/CesarGarech/Biocontrol_modeling_project

```text
@article{AuthorYear,
  author  = {Author Name},
  title   = {Title of Paper},
  journal = {Journal Name},
  year    = {YYYY},
  volume  = {XX},
  number  = {X},
  pages   = {XX-XX},
  doi     = {10.XXXX/YYYY}
}
```
## 🛡 License
This work is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License (or specify your chosen license).
You are free to:

Use, modify, and distribute this code for any purpose.

Cite the appropriate reference when using this code (see References section).
```text
[Author(s) Name].
"[Title of Paper or Project]",
[Journal/Conference Name], vol. [XX], no. [X], [Year].
[DOI: [DOI or URL]]([Link to Paper])
```
See the full license details in the https://www.google.com/search?q=LICENSE file (ensure you have this file).

## 📞 Contact
For any inquiries, please contact César Augusto García Echeverry at cesar.garech@gmail.com or open an issue in this repository.

## ⭐ Acknowledgments
We acknowledge the support of COPPE/UFRJ, [Funding Agency, if any], and all contributors.
