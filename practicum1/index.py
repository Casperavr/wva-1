import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLOCK = True
L_prototype = 117.7 #m
L_pp_prototype = 116.8 #m
B_prototype = 14.3 #m
Diepgang_prototype = 6.2
Wet_surface_prototype = 2261 #m^2
onderwater_volume_prototype = 7952.8 #m^3
C_b_prototype = 0.8591
alpha_prototype_model = 50
v_dienst_prototype = 9.5 * 0.5144 #m/s
v_max_prototype = 10.5 * 0.5144 #m/s
P_prototype = 1300 * 1e3 #W
P_prototype_v_max = 0.7 * P_prototype #W

L_model = L_prototype / alpha_prototype_model

T_zee_C= 12 #C
T_zee_K= T_zee_C + 273.15 #K
T_tank_C= 17 #C
T_tank_K= T_tank_C + 273.15 #K
rho_zee = 1026.6376 #kg/m^3
rho_tank = 998.7780  #kg/m^3
mu_zee = 0.001322 #Pa*s
mu_tank = 0.001080 #Pa*s

mu_kinematisch_zee = mu_zee / rho_zee
mu_kinematisch_tank = mu_tank / rho_tank

def calculateFrictionCoefficient ( Reynolds_nr ):
  C_fm = 0.075 / ( ( np.log10( Reynolds_nr ) - 2)**2)
  return C_fm

def calculateReynoldsnr(v_m, length, mu_kin):
  return v_m * length / mu_kin

def calculateFroude(V_m, L_m): 
  g= 9.81 #m/s
  Fr=  V_m/np.sqrt(g*L_m)
  return Fr

def calculateTotalCoefficient (Resistance, rho, v, s):
  C_tm = Resistance / (0.5 * rho * v**2 * s)
  return C_tm 


#VRAAG 1
v_max_model = v_max_prototype * math.sqrt(1/alpha_prototype_model) #m/s
print('Vraag 1. ', np.round(v_max_model,2), 'm/s')


#VRAAG 2
R_prototype_v_max = P_prototype_v_max / v_max_prototype
print('Vraag 2. ', np.round(R_prototype_v_max,2), 'N')

#VRAAG 3
Reynolds_nr_prototype = calculateReynoldsnr(v_max_prototype, L_prototype, mu_kinematisch_zee)
Reynolds_nr_model = calculateReynoldsnr(v_max_model, L_model, mu_kinematisch_tank)


C_friction_prototype = calculateFrictionCoefficient(Reynolds_nr_prototype)
C_totaal_prototype = R_prototype_v_max / (0.5 * rho_zee * Wet_surface_prototype * v_max_prototype**2)
C_wave_prototype = C_totaal_prototype - C_friction_prototype

C_wave_model = C_wave_prototype
C_friction_model = calculateFrictionCoefficient(Reynolds_nr_model)
C_totaal_model = C_wave_model + C_friction_model
Wet_surface_model = Wet_surface_prototype / (alpha_prototype_model**2)

R_totaal_model = 0.5 * C_totaal_model *  rho_tank * v_max_model**2 * Wet_surface_model
print('Vraag 3. ', R_totaal_model, 'N')


#VRAAG 5
V_m = np.array([0.44, 0.47, 0.51, 0.55, 0.58, 0.62, 0.65, 0.69, 0.72, 0.76, 0.80, 0.84, 0.87, 0.91])
R_tm= np.array([0.64, 0.74, 0.85, 0.97, 1.09, 1.22, 1.36, 1.51, 1.68, 1.85, 2.05, 2.25, 2.48, 2.73])

def verhoudingFroudeFriction(Fr, C_fm):
  return Fr**4/C_fm

def verhoudingFrictionTotal(C_fm, C_tm):
  return C_tm/C_fm


Fr_m = calculateFroude(V_m, L_model)
Re_m = calculateReynoldsnr(V_m, L_model, mu_kinematisch_tank)
C_fm = calculateFrictionCoefficient(Re_m)
C_tm = calculateTotalCoefficient(R_tm, rho_tank, V_m, Wet_surface_model)
fr4_cfm = verhoudingFroudeFriction(Fr_m, C_fm)
ctm_cfm = verhoudingFrictionTotal(C_fm, C_tm)

#VRAAG 6
mask = (Fr_m > 0.1) & (Fr_m < 0.2)
print(fr4_cfm)
x_prohaska = fr4_cfm[mask]   # Fr^4 / C_fm  → x-as
y_prohaska = ctm_cfm[mask]   # C_tm / C_fm  → y-as
print(x_prohaska)


print("Length prohaska points:", len(x_prohaska))

# Lineaire regressie: y = (1+k) + a*x
coeffs = np.polyfit(x_prohaska, y_prohaska, 1)
# coeffs[1] = snijpunt met y-as = (1+k)

form_factor = coeffs[1]  # (1+k)
k = form_factor - 1

# Plot
x_fit = np.linspace(0, max(x_prohaska)*1.1, 100)
y_fit = np.polyval(coeffs, x_fit)

plt.figure(figsize=(8, 5))
plt.scatter(x_prohaska, y_prohaska, color='red', label='Meetdata (Fr < 0.2)')
plt.plot(x_fit, y_fit, color='blue', label=f'Lineaire fit, (1+k) = {form_factor:.3f}')
plt.axvline(x=0, color='gray', linestyle=':')
plt.axhline(y=form_factor, color='green', linestyle='--', label=f'k = {k:.3f}')
plt.xlabel(r'$Fr^4 / C_{FM}$')
plt.ylabel(r'$C_{TM} / C_{FM}$')
plt.title('Prohaska methode voor Formfactor bepaling')
plt.legend()
plt.grid(True)
plt.show(block=BLOCK)

print('Vraag 6. ', k, '-')

# Tabel vraag 5
print(f"\n{'V_m':>8} {'R_tm':>8} {'Fr':>8} {'Re':>12} {'Cfm×10³':>10} {'Ctm×10³':>10} {'Fr⁴/Cfm':>10} {'Ctm/Cfm':>10}")
print("-" * 90)

for i in range(len(V_m)):
  fr  = calculateFroude(V_m[i], L_model)
  re  = calculateReynoldsnr(V_m[i], L_model, mu_kinematisch_tank)
  cfm = calculateFrictionCoefficient(re)
  ctm = calculateTotalCoefficient(R_tm[i], rho_tank, V_m[i], Wet_surface_model)
  fr4_cfm = fr**4 / cfm
  ctm_cfm = ctm / cfm

  print(f"{V_m[i]:>8.2f} {R_tm[i]:>8.2f} {fr:>8.4f} {re:>12.3e} {cfm*1e3:>10.4f} {ctm*1e3:>10.4f} {fr4_cfm:>10.4f} {ctm_cfm:>10.4f}")

#k moet 0.34 zijn oid
# notities en taakverdeling (kan uiteraard nog aangepast worden)

# pagina 12: Casper
# - onderste grafiek y-as verkeerde eenheid (breuk ipv. percentage)
# - bovenste grafiek heeft beetje slecht geformateerd (onderste heftl grafiek is leeg)
# - bovenste grafiek stuwkracht en snelheid controleren (fout in code (vertopt)) snelheid is niet op max snelheid
# pagina 14: Casper
# - gekke hoeken in grafiek, figuur 1.6 tot en met 1.9 (why?)


# pagina 17: Femme
# - bovenste grafiek verkeerd gescaled (veel lege ruimte bovenin)
# - mist extra grafiek met disturbance
# pagina 22: Maaike
# - bovenste grafiek veel te harde snelheid. (fuel rack blijft op 100%) dus een maximum er inzetten. Opdracht te letterlijk genomen, dus wel realistisch blijven. (we zijn niet slim genoeg om te bedenken dat die opdracht niet kan)

# 1.4.4 (staat om meerdere plekken): tijmen
# v3 klopt niet, vlak weerstand en golf weerstand, onderscheid in maken anders verwoorden corrigeren

# pagina 29: Moen
# - cirkel in grafiek enzo klopt niet is raar

# pagina 33: Moen
# * gekke asymptoten in de grafieken oplossen (

# paginaa 35: TIjmen
# - lustfiguur verbeteren naar Lus figuur
# pagina 36: TIjmen
# - conclusie rond v3 beter specificeren "weerstand" betekend hier golf weerstand en niet alle weerstand 
# - subscribt maken laatste regel alinea 4 
# Voor alle grafieken: Femme
# - scale ze naar voledige papier formaat