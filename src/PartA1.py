import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Inputs (single plate fin)

k = 150.0        # W/m·K (aluminum)
h = 100.0      # W/m²·K (side convection)
he = 100.0       # W/m²·K (tip convection)
H = 0.025        # m (fin height in x)
b = 0.001        # m (half-thickness in y)
theta_b = 40.0   # °C (base excess temperature)
N_modes_req = 10 # max modes to seek

# Biot number
Bi = h * b / k
print(f"Biot number (Bi) = {Bi:.6f}\n")


# Eigenproblem: tan(λ b) = h/(k λ)
# bracketing between tan poles

def F(lmbda):
    return np.tan(lmbda * b) - h / (k * lmbda)

def bracket(idx):
    """Interval ((idx-0.5)π/b, (idx+0.5)π/b) contains one root."""
    eps = 1e-12
    a = max(eps, (idx - 0.5) * np.pi / b)
    c = (idx + 0.5) * np.pi / b
    return a, c

lambdas, bounds = [], []
for n in range(N_modes_req):
    a, c = bracket(n)
    try:
        fa = F(a + 1e-9)
        fc = F(c - 1e-9)
        if np.sign(fa) * np.sign(fc) > 0:
            continue  # no sign change → no root in this interval
        root = brentq(F, a + 1e-9, c - 1e-9, maxiter=500, xtol=1e-12, rtol=1e-12)
        lambdas.append(root)
        bounds.append((a, c))
    except ValueError:
        pass

lambdas = np.array(lambdas)

# Modal coefficient α_n = [2 sin(λ b)/λ] / [b + sin(2 λ b)/(2 λ)]
def alpha_n(lam):
    num = 2.0 * np.sin(lam * b) / lam
    den = b + np.sin(2.0 * lam * b) / (2.0 * lam)
    return num / den

# Eigenvalue table 
print("n   lambda [1/m]        alpha_n         LS=tan(λb)       RS=h/(kλ)        LS-RS        lower        upper")
for i, lam in enumerate(lambdas, start=1):
    LS = np.tan(lam * b)
    RS = h / (k * lam)
    lo, up = bounds[i-1]
    print(f"{i:<3} {lam:>12.6f}   {alpha_n(lam):>12.6f}   {LS:>12.6f}   {RS:>12.6f}   {LS - RS:>12.6f}   {lo:>12.6f}   {up:>12.6f}")
print()

# Temperature field (first N modes)

def theta_modes(X, Y, N):
    th = np.zeros_like(X)
    for lam in lambdas[:N]:
        a = alpha_n(lam)
        A = theta_b * a / (np.cosh(lam * H) + (he / (k * lam)) * np.sinh(lam * H))
        th += (A
               * (np.cosh(lam * (H - X)) + (he / (k * lam)) * np.sinh(lam * (H - X)))
               * np.cos(lam * Y))
    return th


# Auto-convergence on modes

Nx, Ny = 100, 70
xv = np.linspace(0.0, H, Nx)
yv = np.linspace(-b, b, Ny)
X, Y = np.meshgrid(xv, yv)

prev = None
converged_N = 1
print("Convergence Check:")
for N in range(1, len(lambdas) + 1):
    cur = theta_modes(X, Y, N)
    if prev is not None:
        rel = np.max(np.abs(cur - prev)) / np.max(np.abs(cur))
        print(f"Change {N-1}→{N}: {rel:.3e}")
        if rel < 1.0e-2:
            converged_N = N
            print(f"\nConverged at N = {N}\n")
            break
    prev = cur
else:
    converged_N = len(lambdas)
    print(f"\nConverged at N = {converged_N} (max modes)\n")

# Final field
theta = theta_modes(X, Y, converged_N)

# Diagnostics
tmax = float(np.max(theta))
tmin = float(np.min(theta))
print(f"Maximum θ: {tmax:.3f} °C")
print(f"Minimum θ: {tmin:.3f} °C")

# Through-thickness variation at base x=0
theta_top = theta_modes(np.array([[0.0]]), np.array([[b]]), converged_N)[0, 0]
theta_ctr = theta_modes(np.array([[0.0]]), np.array([[0.0]]), converged_N)[0, 0]
var_pct = abs(theta_top - theta_ctr) / max(1e-12, abs(theta_ctr)) * 100.0
print(f"Through-thickness variation at x=0: {var_pct:.4f}%")

# Tip center temperature
tip = theta_modes(np.array([[H]]), np.array([[0.0]]), converged_N)[0, 0]
print(f"θ at fin tip center: {tip:.4f} °C")
print(f"Final converged eigenmodes used: {converged_N}")


# Plot

plt.figure(figsize=(8, 4.5))
c = plt.contourf(X * 1000.0, Y * 1000.0, theta, 60, cmap="turbo")
plt.colorbar(c, label="Excess Temperature θ (°C)")
plt.xlabel("x direction (mm)")
plt.ylabel("y direction (mm)")
plt.title(f"2D Temperature Field — {converged_N} Mode Solution\nBi={Bi:.5f}")
plt.tight_layout()
plt.show()
