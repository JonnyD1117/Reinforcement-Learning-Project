import numpy as np
import matplotlib.pyplot as plt
from math import asinh, tanh, cosh
from SPMe_Baseline_Params import SPMe_Baseline_Parameters
from time import time_ns


class Opt_SPMe_Model(SPMe_Baseline_Parameters):
    def __init__(self, init_soc=.5):

        self.SOC_0 = init_soc

        # Initialize Default Parameters
        self.param = {}
        self.param_key_list = ['epsilon_sn', 'epsilon_sp', 'epsilon_e_n', 'epsilon_e_p',
                               'F', 'Rn', 'Rp', 'R', 'T', 'Ar_n', 'Ar_p', 'Ln', 'Lp', 'Lsep', 'Lc',
                               'Ds_n', 'Ds_p', 'De', 'De_p', 'De_n', 'kn', 'kp', 'stoi_n0', 'stoi_n100',
                               'stoi_p0', 'stoi_p100', 'SOC', 'cs_max_n', 'cs_max_p', 'Rf', 'as_n', 'as_p',
                               'Vn', 'Vp', 't_plus', 'cep', 'cen', 'rfa_n', 'rfa_p', 'epsi_sep', 'epsi_e',
                               'epsi_n', 'gamak', 'kappa', 'kappa_eff', 'kappa_eff_sep']
        self.default_param_vals = [self.epsilon_sn, self.epsilon_sp, self.epsilon_e_n, self.epsilon_e_p,
                                   self.F, self.Rn, self.Rp, self.R, self.T, self.Ar_n, self.Ar_p,  self.Ln,
                                   self.Lp, self.Lsep, self.Lc, self.Ds_n, self.Ds_p, self.De, self.De_p,
                                   self.De_n, self.kn, self.kp, self.stoi_n0, self.stoi_n100, self.stoi_p0,
                                   self.stoi_p100, self.SOC, self.cs_max_n, self.cs_max_p, self.Rf, self.as_n,
                                   self.as_p, self.Vn, self.Vp, self.t_plus, self.cep, self.cen, self.rfa_n,
                                   self.rfa_p, self.epsi_sep, self.epsi_e, self.epsi_n, self.gamak, self.kappa,
                                   self.kappa_eff, self.kappa_eff_sep]

        self.param = {self.param_key_list[i]: self.default_param_vals[i] for i in range(0, len(self.param_key_list))}

        # Simulation Settings
        Ts = 1

        # Default Input "Current" Settings
        self.default_current = 25.67            # Base Current Draw

        # Positive electrode three-state state space model for the particle
        self.Ap = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (self.param['Ds_p'] ** 2) / self.param['Rp'] ** 4), - (189 * self.param['Ds_p'] / self.param['Rp'] ** 2)]])
        self.Bp = np.array([[0], [0], [-1]])
        self.Cp = self.param['rfa_p'] * np.array([[10395 * self.param['Ds_p'] ** 2, 1260 * self.param['Ds_p'] * self.param['Rp'] ** 2, 21 * self.param['Rp'] ** 4]])
        self.Dp = np.array([0])

        # Positive electrode SS Discretized
        [n_pos, m_pos] = np.shape(self.Ap)
        self.A_dp = np.eye(n_pos) + self.Ap * Ts
        self.B_dp = self.Bp * Ts
        self.C_dp = self.Cp
        self.D_dp = self.Dp

        # Negative electrode three-state state space model for the particle
        self.An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (self.param['Ds_n'] ** 2) / self.param['Rn'] ** 4), - (189 * self.param['Ds_n'] / self.param['Rn'] ** 2)]])
        self.Bn = np.array([[0], [0], [-1]])
        self.Cn = self.param['rfa_n'] * np.array([[10395 * self.param['Ds_n'] ** 2, 1260 * self.param['Ds_n'] * self.param['Rn'] ** 2, 21 * self.param['Rn'] ** 4]])
        self.Dn = np.array([0])

        # Negative electrode SS Discretized
        [n_neg, m_neg] = np.shape(self.An)
        self.A_dn = np.eye(n_neg) + self.An * Ts
        self.B_dn = self.Bn * Ts
        self.C_dn = self.Cn
        self.D_dn = self.Dn

        # electrolyte  concentration (boundary)
        a_p0 = -(self.param['epsi_n'] ** (3 / 2) + 4 * self.param['epsi_sep'] ** (3 / 2)) / (80000 * self.param['De_p'] * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2))
        b_p0 = (self.param['epsi_n'] ** 2 * self.param['epsi_sep'] + 24 * self.param['epsi_n'] ** 3 + 320 * self.param['epsi_sep'] ** 3 + 160 * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2)) / (19200000000 * (4 * self.param['De_p'] * self.param['epsi_n'] ** (1 / 2) * self.param['epsi_sep'] ** 3 + self.param['De_p'] * self.param['epsi_n'] ** 2 * self.param['epsi_sep'] ** (3 / 2)))

        a_n0 = (self.param['epsi_n'] ** (3 / 2) + 4 * self.param['epsi_sep'] ** (3 / 2)) / (80000 * self.param['De'] * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2))
        b_n0 = (self.param['epsi_n'] ** 2 * self.param['epsi_sep'] + 24 * self.param['epsi_n'] ** 3 + 320 * self.param['epsi_sep'] ** 3 + 160 * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2)) / (19200000000 * (4 * self.param['De_n'] * self.param['epsi_n'] ** (1 / 2) * self.param['epsi_sep'] ** 3 + self.param['De_n'] * self.param['epsi_n'] ** 2 * self.param['epsi_sep'] ** (3 / 2)))

        self.Aep = np.array([[-1 / b_p0, 0], [0, -1 / b_n0]])
        self.Bep = self.param['gamak'] * np.array([[1], [1]])
        self.Cep = np.array([[a_p0 / b_p0, 0], [0, a_n0 / b_n0]])
        self.Dep = np.array([0])

        [n_elec, m] = np.shape(self.Aep)
        self.Ae_dp = np.eye(n_elec) + self.Aep * Ts
        self.Be_dp = self.Bep * Ts
        self.Ce_dp = self.Cep
        self.De_dp = self.Dep

        # sensitivity realization in time domain for epsilon_sp from third order pade(you can refer to my slides)
        coefp = 3 / (self.param['F'] * self.param['Rp'] ** 6 * self.param['as_p'] ** 2 * self.param['Ar_p'] * self.param['Lp'])
        self.Sepsi_A_p = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * self.param['Ds_p'] ** 2) / self.param['Rp'] ** 4, -(189 * self.param['Ds_p']) / self.param['Rp'] ** 2]])
        self.Sepsi_B_p = np.array([[0], [0], [1]])
        self.Sepsi_C_p = coefp * np.array([10395 * self.param['Ds_p'] ** 2, 1260 * self.param['Ds_p'] * self.param['Rp'] ** 2, 21 * self.param['Rp'] ** 4])
        self.Sepsi_D_p = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_p)
        self.Sepsi_A_dp = np.eye(n) + self.Sepsi_A_p * Ts
        self.Sepsi_B_dp = self.Sepsi_B_p * Ts
        self.Sepsi_C_dp = self.Sepsi_C_p
        self.Sepsi_D_dp = self.Sepsi_D_p

        # sensitivity realization in time domain for epsilon_sn from third order pade(you can refer to my slides)
        coefn = 3 / (
                    self.param['F'] * self.param['Rn'] ** 6 * self.param['as_n'] ** 2 * self.param['Ar_n'] * self.param[
                'Ln'])

        self.Sepsi_A_n = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * self.param['Ds_n'] ** 2) / self.param['Rn'] ** 4,
                                                          -(189 * self.param['Ds_n']) / self.param['Rn'] ** 2]])
        self.Sepsi_B_n = np.array([[0], [0], [1]])
        self.Sepsi_C_n = coefn * np.array(
            [10395 * self.param['Ds_n'] ** 2, 1260 * self.param['Ds_n'] * self.param['Rn'] ** 2,
             21 * self.param['Rn'] ** 4])
        self.Sepsi_D_n = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_n)
        self.Sepsi_A_dn = np.eye(n) + self.Sepsi_A_n * Ts
        self.Sepsi_B_dn = self.Sepsi_B_n * Ts
        self.Sepsi_C_dn = self.Sepsi_C_n
        self.Sepsi_D_dn = self.Sepsi_D_n

    def SPMe_step(self, states=None, I_input=None, init_Flag=False):

        epsilon_sn = 0.6  # average negative active volume fraction
        epsilon_sp = 0.50  # average positive active volume fraction
        epsilon_e_n = 0.3  # Liquid [electrolyte] volume fraction (pos & neg)
        epsilon_e_p = 0.3
        F = 96487  # Faraday constant
        Rn = 10e-6  # Active particle radius (pose & neg)
        Rp = 10e-6
        R = 8.314  # Universal gas constant
        T = 298.15  # Ambient Temp. (kelvin)
        Ar_n = 1  # Current collector area (anode & cathode)
        Ar_p = 1
        Ln = 100e-6  # Electrode thickness (pos & neg)
        Lp = 100e-6
        Lsep = 25e-6  # Separator Thickness
        Lc = Ln + Lp + Lsep  # Total Cell Thickness
        Ds_n = 3.9e-14  # Solid phase diffusion coefficient (pos & neg)
        Ds_p = 1e-13
        De = 2.7877e-10  # Electrolyte Diffusion Coefficient
        De_p = De
        De_n = De
        kn = 1e-5 / F  # Rate constant of exchange current density (Reaction Rate) [Pos & neg]
        kp = 3e-7 / F
        # Stoichiometric Coef. used for "interpolating SOC value based on OCV Calcs. at 0.0069% and 0.8228%
        stoi_n0 = 0.0069  # Stoich. Coef. for Negative Electrode
        stoi_n100 = 0.6760
        stoi_p0 = 0.8228  # Stoich. Coef for Positive Electrode
        stoi_p100 = 0.442
        cs_max_n = (3.6e3 * 372 * 1800) / F  # 0.035~0.870=1.0690e+03~ 2.6572e+04
        cs_max_p = (3.6e3 * 274 * 5010) / F  # Positive electrode  maximum solid-phase concentration 0.872~0.278=  4.3182e+04~1.3767e+04
        Rf = 1e-3  #
        as_n = 3 * epsilon_sn / Rn  # Active surface area per electrode unit volume (Pos & Neg)
        as_p = 3 * epsilon_sp / Rp
        Vn = Ar_n * Ln  # Electrode volume (Pos & Neg)
        Vp = Ar_p * Lp
        t_plus = 0.4
        cep = 1000  # Electrolyte Concentration (Assumed Constant?) [Pos & Neg]
        cen = 1000
        SOC_0 = .5
        # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
        rfa_n = 1 / (F * as_n * Rn ** 5)
        rfa_p = 1 / (F * as_p * Rp ** 5)
        epsi_sep = 1
        epsi_e = 0.3
        epsi_n = epsi_e
        gamak = (1 - t_plus) / (F * Ar_n)
        kappa = 1.1046
        kappa_eff = kappa * (epsi_e ** 1.5)
        kappa_eff_sep = kappa * (epsi_sep ** 1.5)

        if init_Flag is True:

            alpha = SOC_0
            stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
            stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant

            xn_old = np.array([[(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ** 2))], [0], [0]])  # stoi_n100 should be changed if the initial soc is not equal to 50 %
            xp_old = np.array([[(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ** 2))], [0], [0]])  # initial positive electrode ion concentration
            xe_old = np.array([[0], [0]])
            Sepsi_p = np.array([[0], [0], [0]])
            Sepsi_n = np.array([[0], [0], [0]])


        else:
            # Unpack Initial battery state variables from dict for use in state space computation
            xn_old = states['xn']
            xp_old = states['xp']
            xe_old = states['xe']
            Sepsi_p = states['Sepsi_p']
            Sepsi_n = states['Sepsi_n']


        # Create Local Copy of Discrete SS Matrices for Ease of notation when writing Eqns.
        A_dp = self.A_dp
        B_dp = self.B_dp
        C_dp = self.C_dp

        A_dn = self.A_dn
        B_dn = self.B_dn
        C_dn = self.C_dn

        Ae_dp = self.Ae_dp
        Be_dp = self.Be_dp
        Ce_dp = self.Ce_dp

        Sepsi_A_dp = self.Sepsi_A_dp
        Sepsi_B_dp = self.Sepsi_B_dp
        Sepsi_C_dp = self.Sepsi_C_dp

        Sepsi_A_dn = self.Sepsi_A_dn
        Sepsi_B_dn = self.Sepsi_B_dn
        Sepsi_C_dn = self.Sepsi_C_dn

        I = I_input

        # Molar Current Flux Density (Assumed UNIFORM for SPM)
        Jn = I / Vn
        Jp = -I / Vp

        if I == 0:
            I = .00000001
            Jn = I / Vn
            Jp = -I / Vp

        # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
        yn_new = C_dn @ xn_old
        yp_new = C_dp @ xp_old
        yep_new = Ce_dp @ xe_old

        # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
        xn_new = A_dn @ xn_old + B_dn * Jn
        xp_new = A_dp @ xp_old + B_dp * Jp
        xe_new = Ae_dp @ xe_old + Be_dp * I

        # Electrolyte Dynamics
        vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F)  # yep(1, k) = positive boundary;

        # Compute "Exchange Current Density" per Electrode (Pos & Neg)
        i_0n = kn * F * (cen * yn_new * (cs_max_n - yn_new)) ** .5
        i_0p = kp * F * (cep * yp_new * (cs_max_p - yp_new)) ** .5

        # Kappa (pos & Neg)
        k_n = Jn / (2 * as_n * i_0n)
        k_p = Jp / (2 * as_p * i_0p)

        # Compute Electrode "Over-potentials"
        eta_n = (R*T*np.log(k_n + (k_n**2 + 1)**0.5))/(F*0.5)
        eta_p = (R*T*np.log(k_p + (k_p**2 + 1)**0.5))/(F*0.5)

        # Record Stoich Ratio (SOC can be computed from this)
        theta_n = yn_new / cs_max_n
        theta_p = yp_new / cs_max_p

        theta = [theta_n, theta_p]

        SOC_n = ((theta_n - stoi_n0) / (stoi_n100 - stoi_n0))
        SOC_p = ((theta_p - stoi_p0) / (stoi_p100 - stoi_p0))
        soc_new = [SOC_n, SOC_p]

        U_n = 0.194 + 1.5 * np.exp(-120.0 * theta_n) + 0.0351 * tanh((theta_n - 0.286) / 0.083) \
              - 0.0045 * tanh((theta_n - 0.849) / 0.119) - 0.035 * tanh((theta_n - 0.9233) / 0.05)\
              - 0.0147 * tanh((theta_n - 0.5) / 0.034) - 0.102 * tanh((theta_n - 0.194) / 0.142) \
              - 0.022 * tanh((theta_n - 0.9) / 0.0164) - 0.011 * tanh((theta_n - 0.124) / 0.0226) \
              + 0.0155 * tanh((theta_n - 0.105) / 0.029)

        U_p = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta_p) + 2.1581 * tanh(52.294 - 50.294 * theta_p) \
              - 0.14169 * tanh(11.0923 - 19.8543 * theta_p) + 0.2051 * tanh(1.4684 - 5.4888 * theta_p) \
              + 0.2531 * tanh((-theta_p + 0.56478) / 0.1316) - 0.02167 * tanh((theta_p - 0.525) / 0.006)

        docv_dCse_n = -1.5 * (120.0 / cs_max_n) * np.exp(-120.0 * theta_n) + (0.0351 / (0.083 * cs_max_n))\
                      * ((cosh((theta_n - 0.286) / 0.083)) ** (-2)) - (0.0045 / (cs_max_n * 0.119))\
                      * ((cosh((theta_n - 0.849) / 0.119)) ** (-2)) - (0.035 / (cs_max_n * 0.05))\
                      * ((cosh((theta_n - 0.9233) / 0.05)) ** (-2)) - (0.0147 / (cs_max_n * 0.034))\
                      * ((cosh((theta_n - 0.5) / 0.034)) ** (-2)) - (0.102 / (cs_max_n * 0.142))\
                      * ((cosh((theta_n - 0.194) / 0.142)) ** (-2)) - (0.022 / (cs_max_n * 0.0164))\
                      * ((cosh((theta_n - 0.9) / 0.0164)) ** (-2)) - (0.011 / (cs_max_n * 0.0226))\
                      * ((cosh((theta_n - 0.124) / 0.0226)) ** (-2)) + (0.0155 / (cs_max_n * 0.029))\
                      * ((cosh((theta_n - 0.105) / 0.029)) ** (-2))


        docv_dCse_p = 0.07645 * (-54.4806 / cs_max_p) * ((1.0 / cosh(30.834 - 54.4806 * theta_p)) ** 2) \
                      + 2.1581 * (-50.294 / cs_max_p) * ((cosh(52.294 - 50.294 * theta_p)) ** (-2)) \
                      + 0.14169 * (19.854 / cs_max_p) * ((cosh(11.0923 - 19.8543 * theta_p)) ** (-2)) \
                      - 0.2051 * (5.4888 / cs_max_p) * ((cosh(1.4684 - 5.4888 * theta_p)) ** (-2)) \
                      - 0.2531 / 0.1316 / cs_max_p * ((cosh((-theta_p + 0.56478) / 0.1316)) ** (-2)) \
                      - 0.02167 / 0.006 / cs_max_p * ((cosh((theta_p - 0.525) / 0.006)) ** (-2))

        # state space Output Eqn. realization for epsilon_s (Neg & Pos)
        y_Sepsi_p = Sepsi_C_dp @ Sepsi_p
        y_Sepsi_n = Sepsi_C_dn @ Sepsi_n

        # state space realization for epsilon_s (Neg & Pos)
        Sepsi_p_new = Sepsi_A_dp @ Sepsi_p + Sepsi_B_dp * I  # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
        Sepsi_n_new = Sepsi_A_dn @ Sepsi_n + Sepsi_B_dn * I

        theta_p_norm = theta_p * cs_max_p
        theta_n_norm = theta_n * cs_max_n

        # rho1p_1 = -np.sign(I) * (-3 * R * T) / (0.5 * F * Rp * as_p) * ((1 + 1 / k_p ** 2) ** (-0.5))
        rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ** 2 + 1) ** 0.5)) * (1 + k_p / ((k_p ** 2 + 1) ** 0.5)) * (-3 * Jp / (2 * as_p ** 2 * i_0p * Rp))
        rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p_norm) / (cep * theta_p_norm * (cs_max_p - theta_p_norm)) * (1 + (1 / (k_p + .00000001) ** 2)) ** (-0.5)

        rho1n = R * T / (0.5 * F) * (1 / (k_n + (k_n ** 2 + 1) ** 0.5)) * (1 + k_n / ((k_n ** 2 + 1) ** 0.5)) * (-3 * Jn / (2 * as_n ** 2 * i_0n * Rn))
        rho2n = (-R * T) / (2 * 0.5 * F) * (cen * cs_max_n - 2 * cen * theta_n_norm) / (cen * theta_n_norm * (cs_max_n - theta_n_norm)) * (1 + 1 / (k_n + .00000001) ** 2) ** (-0.5)

        # sensitivity of epsilon_sp epsilon_sn
        sen_out_spsi_p = (rho1p + (rho2p + docv_dCse_p) * -y_Sepsi_p)
        sen_out_spsi_n = (rho1n + (rho2n + docv_dCse_n) * y_Sepsi_n)
        dV_dEpsi_sp = sen_out_spsi_p
        dV_dEpsi_sn = sen_out_spsi_n

        # dV_dEpsi_sp = sen_out_spsi_p

        # # Surface Concentration Sensitivity for Epsilon (pos & neg)
        # dCse_dEpsi_sp = -1. * y_Sepsi_p * epsi_n
        # dCse_dEpsi_sn = y_Sepsi_p * epsi_n

        V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
        # R_film = -Rf * I / (Ar_n * Ln * as_n)

        states = {'xn': xn_new, 'xp': xp_new, 'xe': xe_new, 'Sepsi_p': Sepsi_p_new, 'Sepsi_n': Sepsi_n_new}
        outputs = {'yn': yn_new, 'yp': yp_new, 'yep': yep_new}
        sensitivity = {'dV_dEpsi_sp': dV_dEpsi_sp, 'dV_dEpsi_sn': dV_dEpsi_sn}

        # if soc_new[1] < .07 or soc_new[0] < .005 or soc_new[1] > 1 or soc_new[0] > 1 or np.isnan(V_term) is True:
        return states, outputs, sensitivity, soc_new, theta, V_term


if __name__ == "__main__":

    model = Opt_SPMe_Model()

    soc_list = []
    sen_list = []
    states = None
    init_time = time_ns()

    step_time_list = []

    for time in range(1800):

        if time == 0:
            init_Flag = True
        else:
            init_Flag = False

        init_step = time_ns()
        new_states, outputs, sensitivity, soc_new, theta, V_term = model.SPMe_step(states, I_input=-25.7, init_Flag=init_Flag)
        post_step = time_ns()

        step_time_list.append((post_step - init_step)*10**-9)

        states = new_states

        soc_list.append(soc_new[0].item())
        sen_list.append(sensitivity['dV_dEpsi_sp'].item())

    print(f"Loop Time: { (time_ns()-init_time)*10**-9 }")

    print(f"Mean Function time: {np.mean(step_time_list)}")

    plt.figure()
    plt.plot(soc_list)
    plt.title('SPMe: SOC Plot')
    plt.xlabel('Time [sec]')
    plt.ylabel('SOC')

    plt.figure()
    plt.plot(sen_list)
    plt.title('SPMe: Active Material Volume Fraction')
    plt.xlabel('Time [sec]')
    plt.ylabel('Epsilon Voltage Sensitivity')
    plt.show()

















