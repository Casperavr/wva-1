# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:13:07 2026

@author: agnes
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 17 2026

@author: agnes

Opdracht 2 - Weerstandsproef WVA1

In dit script wordt de gemeten modelweerstand verwerkt en opgeschaald
naar ware grootte. Hierbij wordt gebruikgemaakt van de ITTC'78-methode
en de Prohaska-methode voor het bepalen van de vormfactor.

Daarnaast wordt de opgeschaalde weerstandskromme vergeleken met de
originele weerstandskromme uit het WVA1-startmodel.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Meetdata weerstandspracticum
# =============================================================================

# Gemeten modelsnelheid [m/s]
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

# Gemeten totale modelweerstand [N]
R_tm_raw = np.array([
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

# Standaarddeviatie van de metingen
# Controleer in het verslag goed of deze waarden inderdaad in N staan.
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


# =============================================================================
# 2. Gegevens van schip, model en water
# =============================================================================

# Gemeten watertemperatuur tijdens practicum
T_water = 19.3                  # [°C]

# Dichtheden
rho_model = 1000                # [kg/m^3] zoet water in sleeptank
rho_ship = 1025                 # [kg/m^3] zeewater

# Kinematische viscositeiten
# Voor nu gebruikt als benadering rond 19-20 graden Celsius.
# In het verslag kun je vermelden dat dit uit watereigenschappen/ITTC-tabellen komt.
nu_model = 1.006e-6             # [m^2/s] zoet water
nu_ship = 1.19e-6               # [m^2/s] zeewater

# Zwaartekracht
g = 9.81                        # [m/s^2]

# Schaalfactor model naar echt schip
lambda_schaal = 50              # [-]

# Gegevens echte Labrax
L_s = 117.70                    # [m] waterlijnlengte
S_s = 2661.0                    # [m^2] nat oppervlak

# Gegevens model
L_m = L_s / lambda_schaal       # [m]
S_m = S_s / lambda_schaal**2    # [m^2]


# =============================================================================
# 3. Nulcorrectie van krachtmeter
# =============================================================================

# Bij nul snelheid hoort de weerstand 0 N te zijn.
# De eerste meting is de nulmeting en wordt daarom gebruikt als offset.
R_offset = R_tm_raw[0]
R_tm_corr = R_tm_raw - R_offset

print("Nulcorrectie")
print("------------")
print(f"Offset krachtmeter = {R_offset:.4f} N")


# =============================================================================
# 4. Plot ruwe en gecorrigeerde meetdata
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(V_m, R_tm_raw, 'o-', label='Ruwe meetdata')
plt.plot(V_m, R_tm_corr, 'o-', label='Gecorrigeerde meetdata')
plt.xlabel(r'Modelsnelheid $V_m$ [m/s]')
plt.ylabel(r'Modelweerstand $R_{tm}$ [N]')
plt.title('Ruwe en gecorrigeerde modelweerstand')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 5. Filteren van bruikbare meetpunten
# =============================================================================

# De nulmeting wordt niet gebruikt, omdat bij de berekening van C_t
# gedeeld wordt door V^2. Een snelheid rond nul geeft dan onbruikbare waarden.
mask = V_m > 0.05

V_m_use = V_m[mask]
R_tm_use = R_tm_corr[mask]
SD_use = SD[mask]


# =============================================================================
# 6. Plot gecorrigeerde meetdata met standaarddeviatie
# =============================================================================

plt.figure(figsize=(8, 5))
plt.errorbar(
    V_m_use,
    R_tm_use,
    yerr=SD_use,
    fmt='o-',
    capsize=4,
    label='Gecorrigeerde meetdata met standaarddeviatie'
)
plt.xlabel(r'Modelsnelheid $V_m$ [m/s]')
plt.ylabel(r'Modelweerstand $R_{tm}$ [N]')
plt.title('Gecorrigeerde modelweerstand met meetspreiding')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 7. Berekeningen op modelschaal
# =============================================================================

# Froudegetal
Fr_m = V_m_use / np.sqrt(g * L_m)

# Reynoldsgetal op modelschaal
Re_m = V_m_use * L_m / nu_model

# ITTC'78 wrijvingsweerstandscoëfficiënt op modelschaal
C_fm = 0.075 / (np.log10(Re_m) - 2)**2

# Totale weerstandscoëfficiënt op modelschaal
C_tm = R_tm_use / (0.5 * rho_model * V_m_use**2 * S_m)


# =============================================================================
# 8. Prohaska-methode voor vormfactor k
# =============================================================================

# Prohaska:
#
# C_tm / C_fm = (1 + k) + a * Fr^4 / C_fm
#
# De snijwaarde met de y-as is dus gelijk aan 1 + k.

x_prohaska = Fr_m**4 / C_fm
y_prohaska = C_tm / C_fm

# Gebruik vooral lage snelheden voor de Prohaska-fit.
# Bij hogere snelheden wordt de golfweerstand groter en is de lineaire
# Prohaska-benadering minder betrouwbaar.
x_fit = x_prohaska
y_fit = y_prohaska

a, b = np.polyfit(x_fit, y_fit, 1)

# b = 1 + k
k = b - 1

print("\nProhaska-resultaten")
print("-------------------")
print(f"Richtingscoëfficiënt a = {a:.4f}")
print(f"Snijpunt b = 1 + k     = {b:.4f}")
print(f"Vormfactor k           = {k:.4f}")


# Plot Prohaska
x_line = np.linspace(0, max(x_prohaska) * 1.05, 100)
y_line = a * x_line + b

plt.figure(figsize=(8, 5))
plt.plot(x_prohaska, y_prohaska, 'o', label='Alle meetpunten')
plt.plot(x_fit, y_fit, 'ro', label='Punten gebruikt voor fit')
plt.plot(x_line, y_line, '-', label='Lineaire Prohaska-fit')
plt.xlabel(r'$Fr^4 / C_{fm}$ [-]')
plt.ylabel(r'$C_{tm} / C_{fm}$ [-]')
plt.title('Prohaska-plot voor bepaling van vormfactor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 9. Opschaling naar ware grootte
# =============================================================================

# Scheepssnelheid via Froude-schaling
V_s = V_m_use * np.sqrt(lambda_schaal)

# Reynoldsgetal op ware grootte
Re_s = V_s * L_s / nu_ship

# ITTC'78 wrijvingsweerstandscoëfficiënt op ware grootte
C_fs = 0.075 / (np.log10(Re_s) - 2)**2

# Golfweerstandscoëfficiënt op modelschaal
C_wm = C_tm - (1 + k) * C_fm

# Door meetspreiding kan C_wm bij lage snelheden negatief worden.
# Golfweerstand hoort fysisch niet negatief te zijn, dus zetten we die waarden op nul.
C_wm_uncorrected = C_wm.copy()
C_wm = np.maximum(C_wm, 0)

# Bij Froude-schaling wordt aangenomen dat de golfweerstandscoëfficiënt gelijk blijft:
C_ws = C_wm

# Viskeuze component op ware grootte
C_viscous = (1 + k) * C_fs

# Totale weerstandscoëfficiënt op ware grootte
C_ts = C_viscous + C_ws

# Totale weerstand op ware grootte
R_ts = 0.5 * rho_ship * V_s**2 * S_s * C_ts


# =============================================================================
# 10. Originele weerstandskromme uit WVA1-startmodel
# =============================================================================

# In het startmodel staat:
#
# c1 = 5.72
# R = (Y_df_set * c1 * v_s^2) * 1000
#
# Voor de vergelijking met de gewone, rustige-water-weerstand nemen we Y = 1.

c1_orig = 5.72
Y_orig = 1.0

R_orig = Y_orig * c1_orig * V_s**2 * 1000


# =============================================================================
# 11. Resultatentabel printen
# =============================================================================

print("\nOpgeschaalde resultaten")
print("-----------------------")
print(
    "V_m [m/s]   V_s [m/s]   V_s [kn]   Fr [-]     "
    "C_ts [-]      R_nieuw [kN]   R_orig [kN]"
)

for i in range(len(V_s)):
    V_s_kn = V_s[i] / 0.5144444444

    print(
        f"{V_m_use[i]:8.3f}   "
        f"{V_s[i]:8.3f}   "
        f"{V_s_kn:8.2f}   "
        f"{Fr_m[i]:8.4f}   "
        f"{C_ts[i]:10.6f}   "
        f"{R_ts[i] / 1000:12.3f}   "
        f"{R_orig[i] / 1000:11.3f}"
    )


# =============================================================================
# 12. Plot opgeschaalde weerstandskromme
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(V_s, R_ts / 1000, 'o-', label='Gemeten weerstand opgeschaald')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel(r'Totale weerstand $R_{ts}$ [kN]')
plt.title('Opgeschaalde weerstandskromme naar ware grootte')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 13. Vergelijking nieuwe en originele weerstandskromme
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(V_s, R_ts / 1000, 'o-', label='Gemeten weerstand opgeschaald')
plt.plot(V_s, R_orig / 1000, 's-', label='Originele weerstand startmodel')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel(r'Totale weerstand $R$ [kN]')
plt.title('Vergelijking originele en opgeschaalde weerstandskromme')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 14. Plot weerstandscomponenten
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(V_s, C_viscous, 'o-', label=r'Viskeuze component $(1+k)C_{fs}$')
plt.plot(V_s, C_ws, 'o-', label=r'Golfcomponent $C_{ws}$')
plt.plot(V_s, C_ts, 'o-', label=r'Totaal $C_{ts}$')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel(r'Weerstandscoëfficiënt [-]')
plt.title('Weerstandscomponenten op ware grootte')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 15. Verhouding van weerstandcomponenten
# =============================================================================

# De verhouding wordt gebruikt om te bespreken welk deel van de totale
# weerstand door viskeuze weerstand en welk deel door golfweerstand komt.

ratio_viscous = C_viscous / C_ts
ratio_wave = C_ws / C_ts

plt.figure(figsize=(8, 5))
plt.plot(V_s, ratio_viscous, 'o-', label='Aandeel viskeuze weerstand')
plt.plot(V_s, ratio_wave, 'o-', label='Aandeel golfweerstand')
plt.xlabel(r'Scheepssnelheid $V_s$ [m/s]')
plt.ylabel(r'Aandeel van totale weerstand [-]')
plt.title('Verhouding van weerstandcomponenten')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 16. Componententabel printen
# =============================================================================

print("\nComponenten weerstand")
print("---------------------")
print(
    "V_s [m/s]   C_viscous [-]   C_wave [-]   C_total [-]   "
    "viscous [%]   wave [%]"
)

for i in range(len(V_s)):
    print(
        f"{V_s[i]:8.3f}   "
        f"{C_viscous[i]:14.6f}   "
        f"{C_ws[i]:10.6f}   "
        f"{C_ts[i]:11.6f}   "
        f"{ratio_viscous[i] * 100:10.1f}   "
        f"{ratio_wave[i] * 100:8.1f}"
    )


# =============================================================================
# 17. Controle negatieve golfweerstand
# =============================================================================

print("\nControle golfweerstandscoëfficiënt")
print("----------------------------------")
print("Ongecorrigeerde negatieve C_wm-waarden worden op 0 gezet.")

for i in range(len(C_wm_uncorrected)):
    if C_wm_uncorrected[i] < 0:
        print(
            f"Bij V_s = {V_s[i]:.3f} m/s was C_wm = "
            f"{C_wm_uncorrected[i]:.6f}; deze is op 0 gezet."
        )