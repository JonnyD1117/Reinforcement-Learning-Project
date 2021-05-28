import numpy as np
import matplotlib.pyplot as plt
from math import asinh, tanh, cosh
from SPMe_Baseline_Params import SPMe_Baseline_Parameters

self.param = {}
self.param_key_list = ['epsilon_sn', 'epsilon_sp', 'epsilon_e_n', 'epsilon_e_p',
                       'F', 'Rn', 'Rp', 'R', 'T', 'Ar_n', 'Ar_p', 'Ln', 'Lp', 'Lsep', 'Lc',
                       'Ds_n', 'Ds_p', 'De', 'De_p', 'De_n', 'kn', 'kp', 'stoi_n0', 'stoi_n100',
                       'stoi_p0', 'stoi_p100', 'SOC', 'cs_max_n', 'cs_max_p', 'Rf', 'as_n', 'as_p',
                       'Vn', 'Vp', 't_plus', 'cep', 'cen', 'rfa_n', 'rfa_p', 'epsi_sep', 'epsi_e',
                       'epsi_n', 'gamak', 'kappa', 'kappa_eff', 'kappa_eff_sep']
self.default_param_vals = [self.epsilon_sn, self.epsilon_sp, self.epsilon_e_n, self.epsilon_e_p,
                           self.F, self.Rn, self.Rp, self.R, self.T, self.Ar_n, self.Ar_p, self.Ln,
                           self.Lp, self.Lsep, self.Lc, self.Ds_n, self.Ds_p, self.De, self.De_p,
                           self.De_n, self.kn, self.kp, self.stoi_n0, self.stoi_n100, self.stoi_p0,
                           self.stoi_p100, self.SOC, self.cs_max_n, self.cs_max_p, self.Rf, self.as_n,
                           self.as_p, self.Vn, self.Vp, self.t_plus, self.cep, self.cen, self.rfa_n,
                           self.rfa_p, self.epsi_sep, self.epsi_e, self.epsi_n, self.gamak, self.kappa,
                           self.kappa_eff, self.kappa_eff_sep]

if custom_params is not None:
    self.param = self.import_custom_parameters(custom_params)

else:
    self.param = {self.param_key_list[i]: self.default_param_vals[i] for i in range(0, len(self.param_key_list))}

def import_custom_parameters(self, new_param_dict):
    if len(self.param) != len(new_param_dict):
        print("New Param Dict is NOT the same dimension as self.param")
        exit(1)
    else:
        key_list = list(self.param)

        self.param = {key_list[i]: new_param_dict[key_list[i]] for i in range(len(self.param))}

    return



@staticmethod
def OCV_Anode(theta):
    # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
    Uref = 0.194 + 1.5 * np.exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh(
        (theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh(
        (theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh(
        (theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)

    return Uref


@staticmethod
def OCV_Cathod(theta):
    Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta) + 2.1581 * tanh(52.294 - 50.294 * theta) - 0.14169 * \
           tanh(11.0923 - 19.8543 * theta) + 0.2051 * tanh(1.4684 - 5.4888 * theta) + 0.2531 * tanh(
        (-theta + 0.56478) / 0.1316) - 0.02167 * tanh((theta - 0.525) / 0.006)

    return Uref



def compute_SOC(self, theta_n, theta_p):
    """
        Computes the value of the SOC from either (N or P) electrode given the current
        Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
    """
    SOC_n = ((theta_n - self.param['stoi_n0'])/(self.param['stoi_n100'] - self.param['stoi_n0']))
    SOC_p = ((theta_p - self.param['stoi_p0'])/(self.param['stoi_p100'] - self.param['stoi_p0']))

    return [SOC_n, SOC_p]

def OCP_Slope_Cathode(self, theta):
    docvp_dCsep = 0.07645 * (-54.4806 / self.param['cs_max_p']) * ((1.0 / cosh(30.834 - 54.4806 * theta)) ** 2) + 2.1581 * (-50.294 / self.param['cs_max_p']) * ((cosh(52.294 - 50.294 * theta)) ** (-2)) + 0.14169 * (19.854 / self.param['cs_max_p']) * ((cosh(11.0923 - 19.8543 * theta)) ** (-2)) - 0.2051 * (5.4888 / self.param['cs_max_p']) * ((cosh(1.4684 - 5.4888 * theta)) ** (-2)) - 0.2531 / 0.1316 / self.param['cs_max_p'] * ((cosh((-theta + 0.56478) / 0.1316)) ** (-2)) - 0.02167 / 0.006 / self.param['cs_max_p'] * ((cosh((theta - 0.525) / 0.006)) ** (-2))

    return docvp_dCsep

def OCP_Slope_Anode(self, theta):
    docvn_dCsen = -1.5 * (120.0 / self.param['cs_max_n']) * np.exp(-120.0 * theta) + (0.0351 / (0.083 * self.param['cs_max_n'])) * ((cosh((theta - 0.286) / 0.083)) ** (-2)) - (0.0045 / (self.param['cs_max_n'] * 0.119)) * ((cosh((theta - 0.849) / 0.119)) ** (-2)) - (0.035 / (self.param['cs_max_n'] * 0.05)) * ((cosh((theta - 0.9233) / 0.05)) ** (-2)) - (0.0147 / (self.param['cs_max_n'] * 0.034)) * ((cosh((theta - 0.5) / 0.034)) ** (-2)) - (0.102 / (self.param['cs_max_n'] * 0.142)) * ((cosh((theta - 0.194) / 0.142)) ** (-2)) - (0.022 / (self.param['cs_max_n'] * 0.0164)) * ((cosh((theta - 0.9) / 0.0164)) ** (-2)) - (0.011 / (self.param['cs_max_n'] * 0.0226)) * ((cosh((theta - 0.124) / 0.0226)) ** (-2)) + (0.0155 / (self.param['cs_max_n'] * 0.029)) * ((cosh((theta - 0.105) / 0.029)) ** (-2))

    return docvn_dCsen