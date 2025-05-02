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
To install and use this project, follow these steps or just run 'run_dashboard.bat':

### Prerequisites
Ensure you have Python installed. Then install the required dependencies:
```
# It's recommended to use a virtual environment
# python -m venv venv
# source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## 📂 Repository Structure

```text
Biocontrol_modeling_project/
├── Body/                   # Core modules: modeling, analysis, estimation, control
│   ├── modeling/           # Bioreactor simulation modes (batch, fed-batch, etc.)
│   ├── analysis.py         # Sensitivity analysis code
│   ├── estimacion_parametros/ # Parameter estimation modules
│   ├── estimation/         # State estimation (EKF) code
│   └── control/            # Regulatory and advanced control modules
├── Utils/                  # Utility functions
│   └── kinetics.py         # Kinetic model definitions
├── Data/                   # Example experimental datasets (.xlsx)
├── Examples/               # Standalone examples (EKF, RTO, NMPC CasADi scripts)
├── LICENSE                 # License information (Add your license file here)
├── main.py                 # Main Streamlit application entry point
├── README.md               # Project documentation (this file)
└── requirements.txt        # Python dependencies list
```

## ✏️ Authors & Contributors

This project was primarily developed by César Augusto García Echeverry at the Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ.

-**César Augusto García Echeverry - Lead Developer - [cesar.garech@gmail.com]**

-**[Contributor 1 Name] - [Role]**

-**[Contributor 2 Name] - [Role]**

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
