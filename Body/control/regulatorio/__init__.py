"""
Regulatory control module.

Implements classical PID (Proportional-Integral-Derivative) control strategies
for bioprocess variables. PID controllers are the foundation of industrial process
control, accounting for over 90% of control loops in practice.

Control loops implemented:
- Temperature control: Heating/cooling jacket manipulation
- pH control: Acid/base addition with split-range control
- Dissolved oxygen (DO) control: Agitation speed or air flow rate manipulation
- Cascade DO control: Nested control loops for improved performance
- On-off feeding control: Binary substrate feeding strategies

Theoretical Background:
PID control law: u(t) = Kc [e(t) + (1/τI)∫e(τ)dτ + τD·de(t)/dt]

Available modules:
- reg_temp: Temperature control
- reg_ph: pH control (split-range)
- reg_oxigeno: Dissolved oxygen control (via agitation)
- reg_cascade_oxigen: Cascade oxygen control
- reg_feed_onoff: On-off feeding control
- reg_ident: pH system identification

References:
- Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic Process Control (3rd ed.). John Wiley & Sons.
- Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). Process Dynamics and Control (4th ed.). John Wiley & Sons.
- Stephanopoulos, G. (1984). Chemical Process Control: An Introduction to Theory and Practice. Prentice Hall.
"""
