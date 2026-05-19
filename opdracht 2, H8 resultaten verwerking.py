# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:17:52 2026

@author: Meike
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Meetdata
# =========================

# gemeten modelsnelheid [m/s]
V_m = np.array([
    -0.0014,
    0.0976,
    0.1972,
    0.2962,
    0.3954,
    0.4946,
    0.5938,
    0.6931,
    0.7923,
    0.9406
])

# gemeten totale modelweerstand [N]
R_tm = np.array([
    -0.2470,
    -0.1982,
    -0.0948,
    0.0565,
    0.2607,
    0.5385,
    0.8519,
    1.2742,
    1.7618,
    2.9575
])

# standaarddeviatie
SD = np.array([
    0.0968,
    1.1826,
    0.7208,
    1.2059,
    1.0227,
    0.9932,
    1.1353,
    2.4482,
    1.3700,
    1.2515
])

# water- en scheepsgegevens

T_water = 19.3              # [°C] watertemperatuur tijdens proef

rho_zoetwater = 1000        # [kg/m^3] dichtheid zoet water
rho_zoutwater = 1025        # [kg/m^3] dichtheid zout water

nu_zoetwater = 1.006e-6     # [m^2/s] kinematische viscositeit zoet water
nu_zoutwater = 1.19e-6      # [m^2/s] kinematische viscositeit zout water

g = 9.81                    # [m/s^2]

lambda_schaal = 50          # [-] schaalfactor

L_s = 117.70                # [m] waterlijnlengte echte schip
S_s = 2661.0                # [m^2] nat oppervlak echte schip

L_m = L_s / lambda_schaal
S_m = S_s / lambda_schaal**2


# Nulcorrectie krachtmeter
# Bij de nulmeting is de weerstand niet 0, maar -0.2470 N.
# Daarom wordt deze waarde van alle metingen afgetrokken.

R_tm_corr = R_tm - R_tm[0]

# Plot gecorrigeerde meetdata

plt.figure(figsize=(8, 5))
plt.plot(V_m, R_tm, 'o-', label='Ruwe meetdata')
plt.plot(V_m, R_tm_corr, 'o-', label='Gecorrigeerde meetdata')
plt.xlabel('Modelsnelheid $V_m$ [m/s]')
plt.ylabel('Modelweerstand $R_{tm}$ [N]')
plt.title('Gemeten modelweerstand met nulcorrectie')
plt.grid()
plt.legend()
plt.show()


# Filteren van nulmeting

# De nulmeting wordt niet gebruikt in de verdere berekeningen,
# omdat hierbij gedeeld wordt door V_m^2.

mask = V_m > 0.05

V_m_use = V_m[mask]
R_tm_use = R_tm_corr[mask]


# Berekeningen op modelschaal

Fr_m = V_m_use / np.sqrt(g * L_m)
Re_m = V_m_use * L_m / nu_zoetwater
C_fm = 0.075 / (np.log10(Re_m) - 2)**2
C_tm = R_tm_use / (0.5 * rho_zoetwater * V_m_use**2 * S_m)


# prohaska-plot en vormfactor

x_prohaska = Fr_m**4 / C_fm
y_prohaska = C_tm / C_fm

a, b = np.polyfit(x_prohaska, y_prohaska, 1)

# b = 1 + k
k = b - 1

print("Vormfactor k =", k)

plt.figure(figsize=(8, 5))
plt.plot(x_prohaska, y_prohaska, 'o', label='Meetpunten')
plt.plot(x_prohaska, a * x_prohaska + b, '-', label='Lineaire fit')
plt.xlabel(r'$Fr^4 / C_{fm}$ [-]')
plt.ylabel(r'$C_{tm} / C_{fm}$ [-]')
plt.title('Prohaska plot')
plt.grid()
plt.legend()
plt.show()


# Opschalen naar ware grootte

V_s = V_m_use * np.sqrt(lambda_schaal)

Re_s = V_s * L_s / nu_zoutwater

C_fs = 0.075 / (np.log10(Re_s) - 2)**2

# golfweerstandscoëfficiënt op modelschaal
C_wm = C_tm - (1 + k) * C_fm

# bij froude-schaling wordt aangenomen
C_ws = C_wm

# totale weerstandscoëfficiënt op ware grootte
C_ts = (1 + k) * C_fs + C_ws

# weerstand op ware grootte
R_ts = 0.5 * rho_zoutwater * V_s**2 * S_s * C_ts


# Resultaten printen

print("\nOpgeschaalde resultaten:")
print("V_s [m/s]     R_ts [N]      C_ts [-]")

for i in range(len(V_s)):
    print(f"{V_s[i]:8.3f}   {R_ts[i]:10.1f}   {C_ts[i]:10.6f}")


# Plot opgeschaalde weerstand
plt.figure(figsize=(8, 5))
plt.plot(V_s, R_ts, 'o-', label='Opgeschaalde weerstand')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel(r'Weerstand $R_{ts}$ [N]')
plt.title('Opgeschaalde weerstandskromme')
plt.grid()
plt.legend()
plt.show()


# Plot weerstandscomponenten
plt.figure(figsize=(8, 5))
plt.plot(V_s, (1 + k) * C_fs, 'o-', label=r'$(1+k)C_{fs}$')
plt.plot(V_s, C_ws, 'o-', label=r'$C_{ws}$')
plt.plot(V_s, C_ts, 'o-', label=r'$C_{ts}$')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel('Weerstandscoëfficiënt [-]')
plt.title('Weerstandscomponenten op ware grootte')
plt.grid()
plt.legend()
plt.show()