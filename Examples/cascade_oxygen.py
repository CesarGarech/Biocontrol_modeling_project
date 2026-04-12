import numpy as np
import matplotlib.pyplot as plt

# --- PID Controller Class ---
# We use a class to create independent instances for each controller
class PIDController:
    """A simple discrete PID controller class."""
    
    def __init__(self, Kp, Ki, Kd, Ts, setpoint=0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.setpoint = setpoint
        
        self.integral = 0.0
        self.prev_error = 0.0
        
        self.min_output, self.max_output = output_limits

    def compute(self, measured_value):
        """Calculates the controller output."""
        error = self.setpoint - measured_value
        
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term
        self.integral += self.Ki * error * self.Ts
        
        # Derivative term
        derivative = self.Kd * (measured_value - self.prev_error) / self.Ts
        
        # Compute output
        output = proportional + self.integral - derivative
        
        # Store current error for next derivative calculation
        self.prev_error = measured_value # Using measured value helps avoid "derivative kick"
        
        # Apply output saturation
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)
            
        return output

    def update_setpoint(self, new_setpoint):
        """Updates the controller's setpoint."""
        self.setpoint = new_setpoint

    def reset(self):
        """Resets the integral and previous error."""
        self.integral = 0.0
        self.prev_error = 0.0

# --- Simulation Parameters ---
Ts = 1.0  # Sample time (s)
t_final = 1000  # Total simulation time (s)
t = np.arange(0, t_final + Ts, Ts)  # Time vector
N = len(t)  # Number of samples

# --- Process Models (Simple First-Order Lags) ---
# We use simple discrete difference equations for simulation
# y[k] = alpha * y[k-1] + (1 - alpha) * K * u[k-1]

# 1. Motor (Inner Loop Process) - FAST
# Input: Power (0-100%), Output: RPM (0-1000)
tau_motor = 10.0  # Time constant (s)
K_motor = 10.0  # Gain (10 RPM per % Power)
alpha_motor = np.exp(-Ts / tau_motor)

# 2. DO Process (Outer Loop Process) - SLOW
# Input: RPM, Output: DO (% Saturation)
tau_do = 80.0  # Time constant (s)
K_do = 0.1  # Gain (0.1 %DO per RPM)
alpha_do = np.exp(-Ts / tau_do)

# 3. Sensors (simple lags)
tau_sensor_rpm = 2.0
alpha_sensor_rpm = np.exp(-Ts / tau_sensor_rpm)
tau_sensor_do = 15.0
alpha_sensor_do = np.exp(-Ts / tau_sensor_do)

# --- Controller Setup ---
# 1. Outer Loop (Master) - DO Controller
# Controls DO by setting RPM_sp
# Tuned to be slower and smoother
pid_DO = PIDController(
    Kp=4.0, Ki=0.05, Kd=0.1, 
    Ts=Ts,
    output_limits=(0, 1000)  # Output is RPM_sp (0-1000 RPM)
)

# 2. Inner Loop (Slave) - RPM Controller
# Controls RPM by setting Motor Power
# Tuned to be fast and aggressive
pid_RPM = PIDController(
    Kp=0.8, Ki=0.2, Kd=0.0,
    Ts=Ts,
    output_limits=(0, 100)  # Output is Power (0-100 %)
)

# --- Initialization of Simulation Arrays ---
DO_sp_vec = np.zeros(N)
DO_actual = np.zeros(N)
DO_meas = np.zeros(N)

RPM_sp_vec = np.zeros(N)  # This is the output of pid_DO
RPM_actual = np.zeros(N)
RPM_meas = np.zeros(N)

Power = np.zeros(N)  # This is the output of pid_RPM
Power_dist = np.zeros(N) # Disturbance on power

# --- Set Initial Conditions ---
DO_actual[0] = 10.0
DO_meas[0] = 10.0
RPM_actual[0] = 100.0
RPM_meas[0] = 100.0

# --- Define Setpoint and Disturbance Profiles ---
DO_sp_vec[t >= 50] = 10.0  # Step up to 30%
DO_sp_vec[t >= 500] = 40.0 # Step down to 20%

# Disturbance: Simulate a 20% drop in motor efficiency (e.g., viscosity increase)
Power_dist[t >= 700] = -20.0

# --- Main Simulation Loop ---
for k in range(1, N):
    
    # --- 1. Setpoint Update ---
    # Update the primary controller's setpoint for this step
    pid_DO.update_setpoint(DO_sp_vec[k])

    # --- 2. Outer Loop (Master) Calculation ---
    # The DO controller computes the *required RPM* based on DO error
    RPM_sp_vec[k] = pid_DO.compute(DO_meas[k-1])
    
    # --- 3. Inner Loop (Slave) Calculation ---
    # Update the secondary controller's setpoint with the master's output
    pid_RPM.update_setpoint(RPM_sp_vec[k])
    
    # The RPM controller computes the *required Power* based on RPM error
    Power[k] = pid_RPM.compute(RPM_meas[k-1])
    
    # --- 4. Process Simulation ---
    
    # Apply disturbance to the final control element
    power_with_disturbance = Power[k] + Power_dist[k]
    power_with_disturbance = np.clip(power_with_disturbance, 0, 100)
    
    # 4a. Inner Process (Motor)
    # RPM_actual depends on the (disturbed) power
    RPM_actual[k] = alpha_motor * RPM_actual[k-1] + \
                    (1 - alpha_motor) * K_motor * power_with_disturbance
    
    # 4b. Outer Process (DO)
    # DO_actual depends on the *actual* RPM
    DO_actual[k] = alpha_do * DO_actual[k-1] + \
                   (1 - alpha_do) * K_do * RPM_actual[k-1]

    # --- 5. Sensor Simulation (with noise) ---
    RPM_meas[k] = alpha_sensor_rpm * RPM_meas[k-1] + \
                  (1 - alpha_sensor_rpm) * RPM_actual[k] + np.random.randn() * 0.5
    
    DO_meas[k] = alpha_sensor_do * DO_meas[k-1] + \
                 (1 - alpha_sensor_do) * DO_actual[k] + np.random.randn() * 0.1

# --- Plot Results ---
plt.figure(figsize=(14, 10))

# Plot 1: Dissolved Oxygen (Outer Loop)
plt.subplot(3, 1, 1)
plt.plot(t, DO_actual, 'b-', label='DO Actual')
plt.plot(t, DO_meas, 'm--', label='DO Measured', alpha=0.7)
plt.plot(t, DO_sp_vec, 'r:', label='DO Setpoint', linewidth=2)
plt.title('Cascade Control: Dissolved Oxygen (Primary Loop)', fontsize=14)
plt.ylabel('% Saturation')
plt.legend()
plt.grid(True)

# Plot 2: Agitation (Inner Loop)
plt.subplot(3, 1, 2)
plt.plot(t, RPM_actual, 'b-', label='RPM Actual')
plt.plot(t, RPM_meas, 'm--', label='RPM Measured', alpha=0.7)
plt.plot(t, RPM_sp_vec, 'g:', label='RPM Setpoint (from DO)', linewidth=2)
plt.title('Agitation (Secondary Loop)', fontsize=14)
plt.ylabel('RPM')
plt.legend()
plt.grid(True)

# Plot 3: Motor Power (Final Output)
plt.subplot(3, 1, 3)
plt.plot(t, Power, 'k-', label='PID Power Output')
plt.plot(t, Power_dist, 'r--', label='Disturbance')
plt.title('Motor Power (Final Control Output)', fontsize=14)
plt.xlabel('Time (s)')
plt.ylabel('Power (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()