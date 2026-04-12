"""
Utils package for bioprocess modeling utilities.

Provides centralized kinetic model definitions for microbial growth and product
formation rates. Models are based on classical bioprocess engineering literature.

Kinetic Models Available:
- Monod model: Simple substrate-limited growth (Monod, 1949)
- Sigmoidal (Hill) model: Cooperative substrate effects
- Complete model: Multiple limiting substrates and product inhibition
- Fermentation model: Mixed aerobic/anaerobic metabolism with Haldane substrate inhibition

All models include CasADi-compatible versions for use in optimization and
advanced control implementations (RTO, NMPC).

References:
- Monod, J. (1949). "The growth of bacterial cultures." Annual Review of Microbiology, 3(1), 371-394.
- Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals (2nd ed.). McGraw-Hill.
- Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall.
- Haldane, J. B. S. (1930). Enzymes. Longmans, Green and Co.
- Andrews, J. F. (1968). "A mathematical model for the continuous culture of microorganisms utilizing 
  inhibitory substrates." Biotechnology and Bioengineering, 10(6), 707-723.
"""

import os
from .kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion