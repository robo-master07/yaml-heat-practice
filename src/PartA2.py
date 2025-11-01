import math
import numpy as np

# Fixed footprint
W = 0.0254
D = 0.0254

# Air (≈40 °C)
rho = 1.184
mu  = 1.85e-5
k_air = 0.026
cp = 1007.0
Ua = 1.0

# Fin material
k_s = 150.0  # Al

R_TARGET = 1.0

# === Notes formulas ===
def Dh(w, L):                     # hydraulic diameter
    return 4*w*L / (2*L + w)

def G_shape(w, L):                # shape parameter
    r = w/L
    return (r*r + 1.0) / ((r + 1.0)*(r + 1.0))

def Nu_fd_lam(G):                 # fully developed laminar
    return 2.055 + 9.614*G

def h_coeff(Nu, Dh):              # h
    return Nu * k_air / Dh

def m_param(h, L, t):             # fin parameter from notes example: m = sqrt(h P / (k A))
    # Plate-fin unit-depth: perimeter P = 2*(t + D), area A = t*D
    return math.sqrt(h*(2*t + 2*D) / (k_s * t * D))

def eta_f_from_mL(mL):            # fin efficiency
    return math.tanh(mL)/mL if mL > 0 else 1.0

def R_fin_single(h, L, t):        # single-fin resistance
    m = m_param(h, L, t)
    return 1.0 / (k_s * D * t * m * math.tanh(m*L))

def R_fh(rho, Ua, L, W, cp):      # fluid heating
    return 1.0 / (2.0 * rho * Ua * L * W * cp)

def Uch_from_notes(Ua, Nf, Nch, t, w):  # EXACT from notes example
    return Ua * ((Nch*w) + (Nf*t)) / (Nch*w)

def sigma_opening(w, t):          # entrance/exit loss ratio
    return w / (w + t)

def f_lam(Re, G):                 # laminar friction factor
    return (4.70 + 19.64*G) / Re

# === sweep ===
best = None
top = []

for L in np.linspace(0.010, 0.040, 61):
    for t in np.linspace(0.0004, 0.0016, 25):
        for Nf in range(10, 80):
            Nch = Nf - 1
            if Nch <= 0: 
                continue
            w = (W - Nf*t) / Nch
            if w <= 0:
                continue

            Dhv = Dh(w, L)
            G    = G_shape(w, L)
            Nu   = Nu_fd_lam(G)
            h    = h_coeff(Nu, Dhv)

            Uch  = Uch_from_notes(Ua, Nf, Nch, t, w)
            Re   = rho * Uch * Dhv / mu
            if Re >= 2300:
                continue  # stay laminar per notes

            # Areas (unit depth)
            A_fs  = 2*L*D*Nf
            A_tip = t*D*Nf
            A_gap = Nch*w*D
            A_tot = A_fs + A_tip + A_gap
            A_fin = A_fs + A_tip

            m  = m_param(h, L, t)
            eta_f = eta_f_from_mL(m*L)
            eta_o = 1.0 - (A_fin/A_tot) * (1.0 - eta_f)

            Rhs = 1.0 / (eta_o * h * A_tot)
            Rfh_val = R_fh(rho, Ua, L, W, cp)
            Rsum = Rhs + Rfh_val

            # pressure losses (optional, per notes)
            sig = sigma_opening(w, t)
            Kc  = -0.4*(sig**2) + 0.5
            Ke  = (sig - 1.0)**2
            f   = f_lam(Re, G)
            q   = 0.5 * rho * Uch**2
            dp  = q * (Kc + 4.0*f*(D/Dhv) + Ke)

            rec = (Rsum, L, t, Nf, w, Dhv, Re, h, eta_f, eta_o, A_tot, Rhs, Rfh_val, dp)
            top.append(rec)

            if Rsum <= R_TARGET and (best is None or L < best[1]):
                best = rec

# === report ===
def line_best(rec):
    Rsum,L,t,Nf,w,Dhv,Re,h,eta_f,eta_o,A_tot,Rhs,Rfh_val,dp = rec
    Nch = Nf - 1
    Uch = Uch_from_notes(Ua, Nf, Nch, t, w)
    sig = sigma_opening(w, t)
    print("\n=== Best design meeting R_sum ≤ 1.0 C/W (notes-only model) ===")
    print(f"Geometry:  L={L:.5f} m, W={W:.5f} m, D={D:.5f} m")
    print(f"           t_fin={t:.5f} m, N_fins={Nf}, N_channels={Nch}")
    print(f"Channel:   w_ch={w:.6f} m, Dh={Dhv:.6f} m, sigma={sig:.3f}")
    print(f"Flow:      Ua={Ua:.3f} m/s, Uch={Uch:.3f} m/s, Re={Re:.1f}")
    print(f"HT:        Nu={Nu_fd_lam(G_shape(w,L)):.2f}, h={h:.1f} W/m^2K")
    print(f"Eff:       eta_f={eta_f:.3f}, eta_o={eta_o:.3f}, A_tot={A_tot:.6f} m^2")
    print(f"Resist:    R_hs={Rhs:.3f} C/W, R_fh={Rfh_val:.3f} C/W, R_sum={Rsum:.3f} C/W")
    print(f"Pressure:  Δp_total≈{dp:.1f} Pa")

if best:
    line_best(best)
    # top 5 by lowest R_sum
    top_sorted = sorted(top, key=lambda x: x[0])[:5]
    print("\n--- Top 5 by lowest R_sum ---")
    for i, r in enumerate(top_sorted, 1):
        Rsum,L,t,Nf,w,Dhv,Re,h,eta_f,eta_o,A_tot,Rhs,Rfh_val,dp = r
        print(f"{i}) L={L:.4f} m, t={t:.4f} m, Nf={Nf}, w={w:.6f} m, Re={Re:.0f}, h={h:.0f}, R_sum={Rsum:.3f}, dp={dp:.0f} Pa")
else:
    print("No geometry met R_sum ≤ 1.0 C/W in this sweep.")
