# -*- coding: UTF-8 -*-
"""
Created by Xiaozhe Wang
        based on Matlab/simulink code of dr. P.deVos
TU Delft 3ME / MTT / SDPO / ME
Version 5.3.1
01-April-2025
History:

20220628: EU: modified plots
20240209: EU: modified parameters for Labrax i.s.o. Visserskotter
20240209: EU: modified function names
20240212: EU: created f_EE function for electrical engine
              modified plot limits/titles
20250401: EU: final cleanup for new course WVA1
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import BDF, LSODA, RK23, RK45, Radau, solve_ivp

#Simulation preference setting

#Initial experiment
Total_time = 36000 #[s] Total simulation time
EXPERIMENT = 5

if EXPERIMENT == 0:
    #ORIGINEEL
    X_iv_t_control = np.array([0,        0.1*Total_time,  0.2*Total_time, 0.5*Total_time, 0.6*Total_time, 0.7*Total_time,  0.8*Total_time, Total_time]) #[s] setting time point
    X_fs    =        np.array([1.0,      1.0,             0.2,            0.2,            0.2,            0.2,             1.0,           1.0        ]) #[-] fuel rack factor, 1 is maximum
    Y_iv_t_control = np.array([0,        0.1*Total_time,  0.2*Total_time, 0.5*Total_time, 0.6*Total_time, 0.7*Total_time,  Total_time  ]) #[s] setting time point
    Y_df    =        np.array([1,        1.0,             1.0,            1.0,            1.0,            1.0,             1.0         ]) #[-] disturbance factor

elif EXPERIMENT == 1:
    #EXPERIMENT 1
    X_iv_t_control = np.array([0,        0.2*Total_time,  0.2*Total_time,  0.4*Total_time, 0.4*Total_time, 0.6*Total_time, 0.6*Total_time, 0.8*Total_time, 0.8*Total_time, Total_time]) #[s] setting time point
    X_fs    =        np.array([1.0,      1.0,             0.75,            0.75,           0.5,            0.5,            0.25,           0.25,           0.01,           0.01      ]) #[-] fuel rack factor, 1 is maximum
    Y_iv_t_control = np.array([0,   Total_time])             #[s] setting time point
    Y_df    =        np.array([1,        1 ])                #[-] disturbance factor

elif EXPERIMENT == 2:
    #EXPERIMENT 2
    X_iv_t_control = np.array([0,        Total_time                     ]) #[s] setting time point
    X_fs    =        np.array([1.0,      1.0                            ]) #[-] fuel rack factor, 1 is maximum
    Y_iv_t_control = np.array([0,        (5/36)*Total_time,   Total_time]) #[s] setting time point
    Y_df    =        np.array([1,        1.5,             1.5,          ]) #[-] disturbance factor

elif EXPERIMENT == 3:
    #EXPERIMENT 3
    dt = 1
    X_iv_t_control = np.arange(dt, Total_time + dt, dt)      #[s] setting time point
    X_fs    =        X_iv_t_control * (19 / Total_time) + 1  #[-] fuel rack factor, 20 is maximum
    Y_iv_t_control = np.array([0,   Total_time])             #[s] setting time point
    Y_df    =        np.array([1,        1 ])                #[-] disturbance factor

elif EXPERIMENT == 4:
    #EXPERIMENT 4
    #Input
    X_fs_amp= 0.5
    X_fs_freq= 0.001
    X_fs_base= 0.5
    X_fs_phase = 0

    dt = 1
    X_iv_t_control = np.arange(dt, Total_time + dt, dt)      #[s] setting time point
    X_fs    =        X_fs_base + X_fs_amp * np.sin(2 * np.pi * X_fs_freq * X_iv_t_control + X_fs_phase)  #[-] fuel rack factor, 20 is maximum apparently
    Y_iv_t_control = np.array([0,   Total_time])             #[s] setting time point
    Y_df    =        np.array([1,        1 ])                #[-] disturbance factor

elif EXPERIMENT == 5:
    #EXPERIMENT 5
    dt = 1
    X_iv_t_control = np.arange(dt, Total_time + dt, dt)      #[s] setting time point
    X_fs    =        X_iv_t_control * (-1 / Total_time) + 1  #[-] fuel rack factor
    Y_iv_t_control = np.arange(dt, Total_time + dt, dt)      #[s] setting time point
    Y_df    =        Y_iv_t_control * (1 / Total_time)       #[-] disturbance factor

def apply_gradual_transition(time_array, factor_array, duration=60):
    """
    Ensures that for every step change in the factor_array, 
    it takes 'duration' seconds to get there.
    """
    new_t = []
    new_f = []
    for i in range(len(time_array)):
        if i > 0 and factor_array[i] != factor_array[i-1]:
            # If the time gap is already larger than duration, 
            # we insert a point to create the slope
            if time_array[i] - time_array[i-1] > duration:
                new_t.append(time_array[i] - duration)
                new_f.append(factor_array[i-1])
        new_t.append(time_array[i])
        new_f.append(factor_array[i])
    return np.array(new_t), np.array(new_f)

# Usage:
X_iv_t_control, X_fs = apply_gradual_transition(X_iv_t_control, X_fs, 60)

#Time step setting
""" 
'var' is variable-time step
'fix' is fixed-time step
"""
time_set = 'var' #Choose fixed-time step or variable-time step
ODE_solver = 'BDF' #Choose ODE solver "RK45, RK23, Radau, BDF, LSODA" 
"""
RK45, RK23 is explicit ode solver. Radau, BDF is implicit ode solver. 
LSODA is for both.
BDF are best choice for WVA1 start model. Other ODE solvers can solve it,
but they have some little problems.
"""
#Fixed-time step setting
time_step = 1 #[s]
Sim_step = int(Total_time / time_step)
time_out = np.linspace(0, Total_time, Sim_step+1)


#Fuel Properties
LHV = 42700 #[kJ/kg]  Lower Heating Value (hoeveelheid energie die vrijkomt als 1kg brandstof wordt verbrand.)
print('fuel properties loaded')

#Physical constant
rho_sea = 1025 #[kg/m^3] density of sea water
print('water properties loaded')

#Ship Data
w = 0.402 # [-] wake factor uit HM sheet Jasper
c1 = 5.72 #6.182 #[-] resistance coefficient of R = c1 * (v_s ^ 2)
thrust_factor = 0.146 #[-] thrust deduction factor uit HM sheet Jasper
m_ship = 8000000 #[kg] ship mass
print('ship data loaded')

#Propeller Data
D_p = 3.3 #[m] propeller diameter
K_Ta = -0.300 #[-] K_T factor a
K_Tb = 0.328  #[-] K_T factor b
K_Qa = -0.045 #[-] K_Q factor a
K_Qb = 0.063  #[-] K_Q factor b
eta_R = 1.01 #[-] relative rotative efficiency
print('propellor data loaded')


# Electrical motor data
# Electro Adda W40 LX6
e_voltage = 600 # V
e_max_current = 1.220E3 # A
e_num_of_engines = 2
e_max_power = 650E3 #W
e_max_torque = 6900 # Nm
e_max_rpm = 900/60 # rps Hz
e_eta_el = 0.888 # from Zuidgeest 3.2.3

eta_e = 0.424 #[-] nominal engine efficiency  from Zuidgeest 3.2.3
print('engine data loaded')

#Gearbox Data
#i_gb = 4.21 #[-] gearbox ratio
i_gb = 6.369 #[-] gearbox ratio
eta_TRM = 0.95 # [-] transmission efficiency
I_tot = 200 # [kg*m^2] total mass of inertia of propulsion system
print('gearbox data loaded')

#Initial Values
v_s0 = 9.2 * 0.5144444444 #[m/s] initial ship speed from verslag S.F.K.Zuidgeest
n_e0 = 900/60 #[rps] Nominal engine speed in rotations from B.1.3. verslag S.F.K.Zuidgeest 
n_p0 = n_e0 / i_gb #[rps] initial rpm
s0 = 0 #[m] travel distance
FC0 = 0 #[g] fuel consumed
P_B0 = 2* 650 #[kW] Nominal electric engine  power from Zuidgeest
M_B0 = P_B0 * 1000 / (2 * math.pi * n_e0) #([P_b*1000/2/math.pi/n_eng_nom])

#Sub function

#Look-up fuelrack
def calc_X_fs_set(t, X_iv_t, Xfs):
    X_fs_set = np.interp(t, X_iv_t, Xfs)
    return X_fs_set

#Look-up disturbance
def f_Y_df_set(t, Y_iv_t, Y_df):
    Y_df_set = np.interp(t, Y_iv_t_control, Y_df)
    return Y_df_set

#Advance Ratio
def calc_AdvanceRatio(v_s, n_p):
    v_a = v_s * (1 - w)
    J = v_a / (n_p * D_p)
    return [v_a, J]

#Propeller Thrust
def f_PTh(v_s, n_p):
    K_T = K_Ta * calc_AdvanceRatio(v_s, n_p)[1] + K_Tb
    F_prop = K_T * (n_p ** 2) * rho_sea * (D_p ** 4)
    return [K_T, F_prop]

#Ship Resistance
def calc_ShipResistance(v_s, Y_df_set):
    R = (Y_df_set * c1 * (v_s ** 2))*1E3
    R_sp = R / (1 - thrust_factor)
    return [R, R_sp]

#Ship Translational Dynamics
def f_STD(v_s, n_p, Y_df_set):
    dv_sdt = (f_PTh(v_s, n_p)[1] - calc_ShipResistance(v_s, Y_df_set)[1]) / m_ship
    dsdt = v_s
    return [dv_sdt, dsdt]

#Propeller Torque
def f_PTo(v_s, n_p):
    K_Q = K_Qa * calc_AdvanceRatio(v_s, n_p)[1] + K_Qb
    Q = K_Q * (n_p ** 2) * rho_sea * (D_p ** 5)
    M_prop = Q / eta_R
    return [K_Q, Q, M_prop]

#Electric Engine
def f_EE(n_p, x_fs_set):
    n_e = n_p * i_gb
    e_current = (x_fs_set * e_max_current)
    W_e = e_voltage * e_current 
    P_B = W_e  * e_num_of_engines * e_eta_el
    M_B = P_B /(2*math.pi * n_e)    
    dFCdt = e_current * e_voltage # vermogen dt sommeert tot electrische energie
    Q_f = (W_e * e_num_of_engines) - P_B
    m_fuel_flux = e_current
    return [n_e, m_fuel_flux, dFCdt, Q_f, W_e, P_B, M_B]

#Shaft Rotational Dynamics
def f_SRD(v_s, n_p, X_fs_set):
    M_TRM = f_EE(n_p, X_fs_set)[6] * i_gb * eta_TRM
    dn_pdt = (M_TRM - f_PTo(v_s, n_p)[2]) / (2 * math.pi * I_tot)
    return [M_TRM, dn_pdt]

#Main program function
#y[0]-v_s  y[1]-n_p y[2]-s y[3]-FC
def main_simulation(t,y):
    X_fs_set = calc_X_fs_set(t, X_iv_t_control, X_fs)
    Y_df_set = f_Y_df_set(t, Y_iv_t_control, Y_df)
    dv_sdt = f_STD(y[0], y[1], Y_df_set)[0] #Ship Translational Dynamics
    dsdt = f_STD(y[0], y[1], Y_df_set)[1] #Ship Translational Dynamics
    dFCdt = f_EE(y[1], X_fs_set)[2] #Diesel Engine
    dn_pdt = f_SRD(y[0], y[1], X_fs_set)[1] #Shaft Rotational Dynamics
    return [dv_sdt, dn_pdt, dsdt, dFCdt]

#ODE solver
if time_set == 'fix':
    #Fixed-time step ODE solver
    sol = solve_ivp(main_simulation, [0, Total_time], [v_s0, n_p0, s0, FC0], method=ODE_solver, t_eval=time_out)
elif time_set == 'var':
    #Variable-time step ODE solver
    sol = solve_ivp(main_simulation, [0, Total_time], [v_s0, n_p0, s0, FC0], method=ODE_solver)
# ik ben cool
#Simulation output
t_out                           = sol.t
v_s_out, n_p_out, s_out, FC_out = sol.y
X_out                           = calc_X_fs_set(sol.t, X_iv_t_control, X_fs)
n_e_out                         = f_EE(n_p_out, X_out)[0]
Y_out                           = f_Y_df_set(sol.t, Y_iv_t_control, Y_df)
R_out                           = calc_ShipResistance(v_s_out, Y_out)[0]
J_out                           = calc_AdvanceRatio(v_s_out, n_p_out)[1]
P_E_out                         = R_out * v_s_out
M_prop_out                      = f_PTo(v_s_out, n_p_out)[2]
P_P_out                         = M_prop_out * 2 * math.pi * n_p_out
M_B_out                         = f_EE(n_p_out, X_out)[6]
P_B_out                         = f_EE(n_p_out, X_out)[5]
R_sp_out                        = calc_ShipResistance(v_s_out, Y_out)[1]
F_prop_out                      = f_PTh(v_s_out, n_p_out)[1]
M_TRM_out                       = f_SRD(v_s_out, n_p_out, X_out)[0]
K_T_out                         = f_PTh(v_s_out, n_p_out)[0]
K_Q_out                         = f_PTo(v_s_out, n_p_out)[0]
v_a_out                         = calc_AdvanceRatio(v_s_out, n_p_out)[0]
P_T_out                         = F_prop_out * v_a_out
Q_out                           = f_PTo(v_s_out, n_p_out)[1]
P_O_out                         = 2 * math.pi * Q_out * n_p_out
# Q_f_out                         = f_EE(n_p_out, X_out)[3]
Q_f_out                         = np.full_like(t_out, f_EE(n_p_out, X_out)[3])
eta_hull_out                    = P_E_out / P_T_out
eta_O_out                       = P_T_out / P_O_out
eta_TRM_out                     = P_P_out / P_B_out
eta_e_out                       = np.linspace(eta_e, eta_e, len(t_out))

print("Time point length:", len(t_out))

#Plot Figure
width = 0.8 #Plot line width setting

plt.figure(figsize=(16,16))
plt.subplot(4, 1, 1)
plt.ylim(0,16)
plt.plot(t_out, v_s_out, linewidth=width)
plt.title('Ship Propulsion Output')
plt.xlabel('Time [s]')
plt.ylabel('Ship speed [m/s]')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t_out, s_out, linewidth=width)
plt.xlabel('Time [s]')
plt.ylabel('Distance travelled [m]')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t_out, FC_out, linewidth=width)
plt.xlabel('Time [s]')
plt.ylabel('Electric energy consumed [Ws=J]')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.ylim(0,2)
plt.plot(t_out, X_out, linewidth=width)
plt.xlabel('Time [s]')
plt.ylabel('Fuel rack [%]')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'figures/start_fig01_EXPERIMENT-{EXPERIMENT}.jpg')


plt.figure(2, figsize=(16,9))
plt.ylim(0, 16)
plt.plot(t_out, v_s_out, t_out, n_e_out, t_out, n_p_out, linewidth=width)
plt.legend(('Ship speed [m/s]', 'Engine rot. speed [rps]', 'Propeller rot. speed [rps]'), loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('Ship speed [m/s], Engine / Propeller speed [rps]')
plt.grid(True)
plt.title('figuur 2', loc='right', pad=20)
plt.savefig(f'figures/Speeds-Time_EXPERIMENT-{EXPERIMENT}.jpg')

def plotLineGraph(x, ylist, xlabel, ylabels, title, yunit):
    plt.figure(figsize=(16,9))
    for i in range(len(ylist)):
        plt.plot(x, ylist[i], linewidth=width, label=ylabels[i])
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(yunit)
    plt.grid(True)
    plt.savefig(f'figures/{title.replace(" ", "")}_EXPERIMENT-{EXPERIMENT}.jpg')    
    plt.show(block = False)
    
plotLineGraph(v_s_out, [R_out], 'Ship speed [m/s]', ["Resistance [N]"], "Ship resistance - Ship speed", "[N]")
plotLineGraph(v_a_out, [F_prop_out], 'Instroomsnelheid bij schroef [m/s]', ["Schroefstuwkracht [N]"], "Instroomsnelheid - Schroefstuwkracht", "[N]")
plotLineGraph(x=n_p_out, ylist=[M_prop_out], xlabel='Propeller rot. speed [rps]', ylabels=["Schroefkoppel [Nm]"], title="Propeller rot. speed - Schroefkoppel", yunit="[Nm]")
plotLineGraph(x=n_e_out, ylist=[M_B_out], xlabel='Engine rot. speed [rps]', ylabels=["Elektromotorkoppel [Nm]"], title="Engine rot. speed - Elektromotorkoppel",yunit="[Nm]")
plotLineGraph(x=t_out, ylist=[P_E_out, P_O_out, P_B_out, Q_f_out], xlabel='Time [s]', ylabels=["Effectief vermogen [W]", "Geleverd vermogen [W]", "Elektromotor vermogen [W]", "Warmteverlies elektromotor [W]"], title="Vermogens - Tijd", yunit="[W]")
plotLineGraph(x=t_out, ylist=[eta_hull_out, eta_O_out, eta_TRM_out, eta_e_out], xlabel='Time [s]', ylabels=["Romprendement [-]", "Open-water schroefrendement [-]", "Transmissie-efficiëntie [-]", "Nominaal elektrisch motorrendement [-]"], title="Efficiënties - Tijd", yunit="[-]")
plotLineGraph(x=P_P_out, ylist=[eta_O_out], xlabel='Propellor vermogen [W]', ylabels=["Open-water schroefrendement [-]"], title="Propellor vermogen - schroefrendement",yunit="[-]")
plotLineGraph(x=P_B_out, ylist=[eta_e_out], xlabel='Elektromotor vermogen [W]', ylabels=["Transmissie-efficiëntie [-]"], title="Transmissie-efficiëntie - schroefrendement",yunit="[-]")

plt.show(block=False)
print('End simulation run')