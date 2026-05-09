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
- **Digital Twin for Distillation Columns:** 
  - DWSIM integration for rigorous distillation simulation
  - SCADA data generation and analysis with outlier detection (IQR), moving average filtering, and WLS reconciliation
  - Machine Learning prediction of ethanol composition using multiple models (Neural Networks, Random Forest, Decision Tree, SVR, Gradient Boosting) with comprehensive performance comparison
- **Interactive User Interface:** Built with Streamlit (`main.py`) for easy navigation and visualization using Matplotlib.
- **AI Guide (Beta):** Optional LLM-powered assistant using Ollama for contextual help with equations, methods, parameters, and references.

## 🤖 AI Guide: LLM Assistant (Ollama-based) - **NOW IMPLEMENTED**
The AI Guide is now available as an optional feature to assist with bioprocess modeling concepts.

### ✅ Implemented Features
- **Contextual Help:** Explains equations and methods based on the current page
- **Parameter Suggestions:** Provides typical ranges from literature
- **Reference Recommendations:** Curated bibliographic references for each topic
- **Free & Local:** Uses Ollama (no API costs, data stays local)
- **Graceful Fallback:** App works normally if Ollama is unavailable

### 🚀 Quick Start Guide

#### 1. Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

#### 2. Download a model
```bash
# Recommended: Fast and capable
ollama pull llama3.1:8b

# Alternatives:
ollama pull qwen2.5:7b    # Good multilingual support
ollama pull llama3.2:3b   # Smaller, faster
```

#### 3. Start Ollama server
```bash
ollama serve
```

#### 4. Enable in the app
- Open the application (`streamlit run main.py`)
- In the sidebar, find **"🤖 AI Guide (Beta)"**
- Check **"Activar Asistente IA"**
- Click **"🔍 Verificar Conexión"** to test

### 💡 Usage Examples
- **Explain equations:** Click "📖 Explicar método" to understand the current model
- **Parameter help:** Click "📊 Sugerir parámetros" for typical ranges
- **Custom questions:** Type in the text area, e.g., "¿Qué significa Ks en Monod?"

### ⚙️ Configuration
- **Model selection:** Choose from available Ollama models in settings
- **Custom URL:** Configure if Ollama runs on a different port/host
- **Chat history:** Last 3 interactions are preserved in the sidebar

### ⚠️ Important Notes
- **Educational use only:** Suggestions are orientative and require experimental validation
- **No internet needed:** All processing is local (if using local Ollama)
- **Model size:** Smaller models (3B-7B) are faster; larger models (13B+) are more capable
- **References:** Only curated academic references from project documentation are provided

## 📦 Installation

### ⚠️ Important: Python Version Requirement

**This project requires Python 3.10.x (specifically 3.10.9 or higher patch versions within 3.10).** Newer versions of Python (3.11+) have compatibility issues with some of the libraries used in this project, particularly:
- **TensorFlow**: Requires specific Python version compatibility for optimal performance
- **CasADi**: Has known issues with newer Python versions due to C++ binding changes
- **NumPy/SciPy**: Version conflicts can arise with newer Python releases

The `run_dashboard.bat` (Windows) or `run_dashboard.sh` (Linux/Mac) script will automatically verify your Python version and guide you through installing Python 3.10.x if needed.

### Option 1: Quick Start (Windows)
Just run the provided batch script:
```bash
run_dashboard.bat
```
This will automatically:
1. Verify you have Python 3.10.x installed
2. Guide you through installation if needed
3. Create a virtual environment
4. Install dependencies
5. Launch the dashboard

### Option 1b: Quick Start (Linux/Mac)
Just run the provided shell script:
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```
This will automatically:
1. Verify you have Python 3.10.x installed
2. Guide you through installation if needed
3. Create a virtual environment
4. Install dependencies
5. Launch the dashboard

### Option 2: Manual Installation
Ensure you have Python 3.10.x (3.10.14 or higher patch version) installed. Then:

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
│   ├── digital_twin/       # Digital Twin for distillation columns
│   │   ├── __init__.py
│   │   ├── gemelo_digital.py        # Main router for Digital Twin
│   │   ├── simulacion_dwsim.py      # DWSIM simulation interface
│   │   ├── analisis_datos.py        # SCADA data analysis and reconciliation
│   │   └── ml_prediction.py         # Machine Learning prediction models
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


## 📚 Theoretical Background

This project is built upon well-established theoretical foundations in bioprocess engineering, process control, optimization, and numerical methods. The following sections provide key references for understanding the underlying theory.

### Bioprocess Engineering and Modeling

**Kinetic Models and Microbial Growth:**
- **Shuler, M. L., & Kargi, F. (2002).** *Bioprocess Engineering: Basic Concepts* (2nd ed.). Prentice Hall. 
  - Comprehensive coverage of microbial kinetics including Monod, Haldane, and substrate/product inhibition models
  - Mass balance equations for batch, fed-batch, and continuous bioreactors
  
- **Bailey, J. E., & Ollis, D. F. (1986).** *Biochemical Engineering Fundamentals* (2nd ed.). McGraw-Hill.
  - Foundational text on bioprocess modeling and stoichiometry
  - Detailed treatment of oxygen transfer (kLa) and metabolic pathways

- **Luedeking, R., & Piret, E. L. (1959).** "A kinetic study of the lactic acid fermentation. Batch process at controlled pH." *Journal of Biochemical and Microbiological Technology and Engineering*, 1(4), 393-412.
  - Classic paper establishing the Luedeking-Piret model for product formation (qP = α·μ + β)

### Classical Process Control

**PID Control and Regulatory Strategies:**
- **Smith, C. A., & Corripio, A. B. (2005).** *Principles and Practice of Automatic Process Control* (3rd ed.). John Wiley & Sons.
  - Standard reference for PID controller design and tuning
  - Cascade control, split-range control, and feedforward strategies
  - Chapters on controller tuning methods (Ziegler-Nichols, Cohen-Coon, etc.)

- **Stephanopoulos, G. (1984).** *Chemical Process Control: An Introduction to Theory and Practice*. Prentice Hall.
  - Comprehensive treatment of feedback and feedforward control
  - Process dynamics, transfer functions, and frequency response

- **Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016).** *Process Dynamics and Control* (4th ed.). John Wiley & Sons.
  - Modern perspective on process control with extensive examples
  - Digital control implementation and discrete-time systems

### Advanced Process Control

**Model Predictive Control (MPC/NMPC):**
- **Camacho, E. F., & Bordons, C. (2007).** *Model Predictive Control* (2nd ed.). Springer-Verlag.
  - Comprehensive coverage of linear and nonlinear MPC formulations
  - Constraint handling and optimization problem formulation

- **Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017).** *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
  - Advanced treatment of MPC theory including stability and robustness
  - Nonlinear MPC and moving horizon estimation

**Real-Time Optimization (RTO):**
- **Biegler, L. T. (2010).** *Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes*. SIAM.
  - Mathematical foundations for RTO and dynamic optimization
  - IPOPT and other NLP solvers
  
- **Marlin, T. E. (2000).** *Process Control: Designing Processes and Control Systems for Dynamic Performance* (2nd ed.). McGraw-Hill.
  - Integration of process design and control
  - Real-time optimization in the context of process operations

**Fuzzy Logic Control:**
- **Zadeh, L. A. (1965).** "Fuzzy sets." *Information and Control*, 8(3), 338-353.
  - Foundational paper introducing fuzzy set theory
  
- **Mamdani, E. H., & Assilian, S. (1975).** "An experiment in linguistic synthesis with a fuzzy logic controller." *International Journal of Man-Machine Studies*, 7(1), 1-13.
  - First application of fuzzy logic to control systems
  
- **Passino, K. M., & Yurkovich, S. (1998).** *Fuzzy Control*. Addison Wesley Longman.
  - Comprehensive textbook on fuzzy control theory and applications
  
- **Wang, L. X. (1997).** *A Course in Fuzzy Systems and Control*. Prentice Hall PTR.
  - Detailed treatment of fuzzy systems design and implementation
  
- **Ross, T. J. (2010).** *Fuzzy Logic with Engineering Applications* (3rd ed.). John Wiley & Sons.
  - Practical applications of fuzzy logic in engineering
  
- **Lee, C. C. (1990).** "Fuzzy logic in control systems: fuzzy logic controller-Parts I and II." *IEEE Transactions on Systems, Man, and Cybernetics*, 20(2), 404-435.
  - Comprehensive review of fuzzy logic controllers
  
- **Chen, L., et al. (1996).** "Fuzzy logic based control of dissolved oxygen in a bioprocess." *Computers & Chemical Engineering*, 20, S1337-S1342.
  - Application to bioprocess control

### State Estimation

**Extended Kalman Filter (EKF) and Observers:**
- **Jazwinski, A. H. (1970).** *Stochastic Processes and Filtering Theory*. Academic Press.
  - Theoretical foundation of Kalman filtering for nonlinear systems

- **Simon, D. (2006).** *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*. John Wiley & Sons.
  - Practical treatment of EKF implementation
  - Application to nonlinear bioprocess systems

- **Bastin, G., & Dochain, D. (1990).** *On-line Estimation and Adaptive Control of Bioreactors*. Elsevier.
  - Specialized reference for bioprocess state estimation
  - Asymptotic observers and adaptive control

### Numerical Methods and Optimization

**Ordinary Differential Equations (ODEs):**
- **Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).** *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
  - Implementation of Runge-Kutta and other ODE solvers
  - Used by scipy.integrate.solve_ivp

**Optimization Algorithms:**
- **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization* (2nd ed.). Springer.
  - Comprehensive coverage of optimization theory
  - Levenberg-Marquardt, Sequential Quadratic Programming (SQP), and interior-point methods

- **Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., & Diehl, M. (2019).** "CasADi: a software framework for nonlinear optimization and optimal control." *Mathematical Programming Computation*, 11(1), 1-36.
  - CasADi framework used for NMPC and RTO in this project

### Statistical Methods

**Parameter Estimation and Confidence Intervals:**
- **Bard, Y. (1974).** *Nonlinear Parameter Estimation*. Academic Press.
  - Theory and methods for fitting kinetic parameters to experimental data
  
- **Beck, J. V., & Arnold, K. J. (1977).** *Parameter Estimation in Engineering and Science*. John Wiley & Sons.
  - Statistical analysis of parameter estimates
  - Confidence intervals and sensitivity coefficients

**Design of Experiments:**
- **Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005).** *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). John Wiley & Sons.
  - Experimental design for bioprocess optimization

### Software and Tools

- **SciPy:** Virtanen, P., et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods*, 17(3), 261-272.
- **NumPy:** Harris, C. R., et al. (2020). "Array programming with NumPy." *Nature*, 585(7825), 357-362.
- **Streamlit:** Used for interactive dashboard development
- **Matplotlib:** Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment." *Computing in Science & Engineering*, 9(3), 90-95.

## 🔬 Publications & Citations
If you use this work in your research, please cite the following:

GitHub Repository: https://github.com/CesarGarech/Biocontrol_modeling_project

```text
@software{Garcia2024Biocontrol,
  author  = {García Echeverry, César Augusto and Franco Reyes, Andrés Mateo and Vélez Orjuela, Juan Pablo and Gámez Herazo, Carolina del Mar},
  title   = {Biocontrol Modeling Project: Interactive Tools for Bioprocess Control Education},
  year    = {2024},
  url     = {https://github.com/CesarGarech/Biocontrol_modeling_project},
  institution = {LADES - COPPE/UFRJ}
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
