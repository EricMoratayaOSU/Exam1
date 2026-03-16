def f1(x, y1, y2):
    """Derivative of y1 (which is y')"""
    return y2

def f2(x, y1, y2):
    """Derivative of y2 (which is y'')"""
    return y1 + x

def improved_euler(y0, yp0, h, target_x):
    """Solves the system using the Improved Euler (Heun's) method."""
    n_steps = int(round(target_x / h))
    x = 0.0
    y1 = y0
    y2 = yp0
    
    for _ in range(n_steps):
        # Predictor step
        k1_1 = f1(x, y1, y2)
        k1_2 = f2(x, y1, y2)
        
        u1 = y1 + h * k1_1
        u2 = y2 + h * k1_2
        
        # Corrector step
        k2_1 = f1(x + h, u1, u2)
        k2_2 = f2(x + h, u1, u2)
        
        y1 = y1 + (h / 2.0) * (k1_1 + k2_1)
        y2 = y2 + (h / 2.0) * (k1_2 + k2_2)
        
        x += h
        
    return y1, y2

def runge_kutta_4(y0, yp0, h, target_x):
    """Solves the system using the 4th-order Runge-Kutta method."""
    n_steps = int(round(target_x / h))
    x = 0.0
    y1 = y0
    y2 = yp0
    
    for _ in range(n_steps):
        k1_1 = f1(x, y1, y2)
        k1_2 = f2(x, y1, y2)
        
        k2_1 = f1(x + h/2, y1 + (h/2)*k1_1, y2 + (h/2)*k1_2)
        k2_2 = f2(x + h/2, y1 + (h/2)*k1_1, y2 + (h/2)*k1_2)
        
        k3_1 = f1(x + h/2, y1 + (h/2)*k2_1, y2 + (h/2)*k2_2)
        k3_2 = f2(x + h/2, y1 + (h/2)*k2_1, y2 + (h/2)*k2_2)
        
        k4_1 = f1(x + h, y1 + h*k3_1, y2 + h*k3_2)
        k4_2 = f2(x + h, y1 + h*k3_1, y2 + h*k3_2)
        
        y1 = y1 + (h / 6.0) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        y2 = y2 + (h / 6.0) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        
        x += h
        
    return y1, y2

def main():
    print("For the initial value problem y''- y = x")
    
    # Get initial inputs, with fallbacks to defaults if the user just hits Enter
    y0_str = input("Enter the value of y at x=0: ").strip()
    y0 = float(y0_str) if y0_str else 1.0
    
    yp0_str = input("Enter the value of y' at x=0: ").strip()
    yp0 = float(yp0_str) if yp0_str else -2.0
    
    h_str = input("Enter the step size for the numerical solution: ").strip()
    h = float(h_str) if h_str else 0.1
    
    while True:
        target_x = float(input("At what value of x do you want to know y and y'? "))
        
        y_ie, yp_ie = improved_euler(y0, yp0, h, target_x)
        y_rk, yp_rk = runge_kutta_4(y0, yp0, h, target_x)
        
        print(f"\nAt x={target_x:.3f}")
        print(f"For the improved Euler method: y={y_ie:.3f}, and y'={yp_ie:.3f}")
        print(f"For the Runge-Kutta method: y={y_rk:.3f}, and y'={yp_rk:.3f}")
        
        again = input("\nDo you want to compute at a different x? (Y/N) ").strip().lower()
        if not again.startswith('y'):
            break

if __name__ == "__main__":
    main()