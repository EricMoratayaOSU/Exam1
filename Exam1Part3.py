import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

class circuit:
    def __init__(self, R, L, C, amplitude, frequency, phase):
        self.R = R
        self.L = L
        self.C = C
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        
        # Arrays to hold simulation results
        self.t = None
        self.i_L = None
        self.v_c = None
        self.i_1 = None
        self.i_2 = None

    def v(self, t):
        """Input voltage function v(t)"""
        return self.amplitude * np.sin(self.frequency * t + self.phase)

    def ode_system(self, t, state):
        """
        System of differential equations in state form.
        state[0] = i_L (current through inductor)
        state[1] = v_c (voltage across capacitor)
        """
        i_L, v_c = state
        
        # di_L/dt = (v(t) - v_c) / L
        diL_dt = (self.v(t) - v_c) / self.L
        
        # dv_c/dt = i_L/C - v_c/(R*C)
        dvc_dt = (i_L / self.C) - (v_c / (self.R * self.C))
        
        return [diL_dt, dvc_dt]

    def simulate(self):
        """Simulate the circuit for 10 seconds using solve_ivp."""
        t_span = (0, 10)
        # Generate enough points for a smooth plot of a 20 rad/s sine wave
        t_eval = np.linspace(0, 10, 2000) 
        
        # Initial conditions: 0 current, 0 voltage
        initial_conditions = [0.0, 0.0]
        
        sol = solve_ivp(self.ode_system, t_span, initial_conditions, t_eval=t_eval)
        
        self.t = sol.t
        self.i_L = sol.y[0]
        self.v_c = sol.y[1]
        
        # Calculate i1 (through R) and i2 (through C)
        self.i_1 = self.v_c / self.R
        self.i_2 = self.i_L - self.i_1

    def doPlot(self):
        """Plot the currents and voltage exactly as specified."""
        fig, ax1 = plt.subplots(figsize=(10, 7))
        
        # Plot currents on the left y-axis
        ax1.plot(self.t, self.i_1, 'k-', label='$i_1(t)$')
        ax1.plot(self.t, self.i_2, 'k--', label='$i_2(t)$')
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('$i_1, i_2$ (A)')
        ax1.set_xlim(0, 10)
        
        # Set up the grid to match the image
        ax1.grid(True, which='major', linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper right')
        
        # Plot voltage on the right y-axis (twinx)
        ax2 = ax1.twinx()
        ax2.plot(self.t, self.v_c, 'k:', label='$v_c(t)$')
        ax2.set_ylabel('$v_c(t)$ (V)')
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()

def get_input(prompt, default_value):
    """Helper function to get user input with a default fallback."""
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    if user_input == "":
        return default_value
    return float(user_input)

def main():
    print("RLC Circuit Simulator")
    
    while True:
        # Solicit parameters from user
        R = get_input("Enter R (Ohms)", 10.0)
        L = get_input("Enter L (H)", 20.0)
        C = get_input("Enter C (F)", 0.05)
        amp = get_input("Enter Amplitude (V)", 20.0)
        freq = get_input("Enter Frequency (rad/s)", 20.0)
        phase = get_input("Enter Phase (rad)", 0.0)
        
        # Instantiate circuit, simulate, and plot
        my_circuit = circuit(R, L, C, amp, freq, phase)
        print("\nRunning simulation...")
        my_circuit.simulate()
        my_circuit.doPlot()
        
        # Prompt to run again
        run_again = input("\nChange parameters and simulate again? (y/n) [n]: ").strip().lower()
        if not run_again.startswith('y'):
            print("Exiting program.")
            break

if __name__ == "__main__":
    main()