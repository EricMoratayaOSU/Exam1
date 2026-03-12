import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# differential equation
def dydx(x, y):
    return (y - 0.01 * x**2)**2 * np.sin(x**2) + 0.02 * x

# The exact solution
def exact_solution(x):
    # S(x) calculated using quad
    S_x, _ = quad(lambda t: np.sin(t**2), 0, x)
    return 1 / (2.5 - S_x) + 0.01 * x**2

def main():
    x_span = (0, 5)
    y0 = [0.4]
    
    # h = 0.2 steps
    x_eval = np.arange(0, 5.2, 0.2) 
    
    # x array
    x_exact = np.linspace(0, 5, 200)

    # Calculating solutions
    sol = solve_ivp(dydx, x_span, y0, t_eval=x_eval)
    y_exact = [exact_solution(x) for x in x_exact]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Plot exact (solid line) and numerical (triangles)
    ax.plot(x_exact, y_exact, 'k-', label='Exact')
    ax.plot(sol.t, sol.y[0], '^', color='black', markerfacecolor='none', linestyle='none', label='Numerical')

    # 2 & 3. Set limits and tick parameters
    ax.set_xlim([0.0, 6.0])
    ax.set_ylim([0.0, 1.0])
    ax.tick_params(axis='x', direction='in', top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', left=True, right=True)

    # 5. Format numbers on axes to one decimal digit
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 4. Legend
    ax.legend(loc='upper right')

    # 6. Title (matching the typo '+ 0.2x' from the screenshot exactly)
    ax.set_title("IVP: $y' = (y - 0.01x^2)^2sin(x^2) + 0.2x, y(0) = 0.4$")

    plt.show()

if __name__ == "__main__":
    main()