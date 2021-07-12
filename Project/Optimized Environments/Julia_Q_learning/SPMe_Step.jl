
module SPMe_Battery_Model

export SPMe_step


function SPMe_step(xn_old=nothing, xp_old=nothing, xe_old=nothing, Sepsi_p=nothing, Sepsi_n=nothing, I_input=nothing, init_flag=false)
    #=
    """
    step function runs one iteration of the model given the input current and returns output states and quantities
    States: dict(), I_input: scalar, state_of_charge: scalar
    """
    =#
    SOC_0 = .5

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

    SOC = 1  # SOC can change from 0 to 1

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

    # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
    rfa_n = 1 / (F * as_n * Rn ^ 5)
    rfa_p = 1 / (F * as_p * Rp ^ 5)

    epsi_sep = 1
    epsi_e = 0.3
    epsi_n = epsi_e
    gamak = (1 - t_plus) / (F * Ar_n)

    kappa = 1.1046
    kappa_eff = kappa * (epsi_e ^ 1.5)
    kappa_eff_sep = kappa * (epsi_sep ^ 1.5)

    ####################################################################################################################
    ####################################################################################################################
    # Simulation Settings
    Ts = 1
    simulation_time = 3600
    num_steps = simulation_time // Ts

    # Default Input "Current" Settings
    default_current = 25.67  # Base Current Draw
    ###################################################################################################################
    ###################################################################################################################

    if init_flag == true || xn_old == nothing || xp_old == nothing || xe_old == nothing || Sepsi_p == nothing || Sepsi_n == nothing

        # Initialize the "battery" and 'sensitivity' states (FOR STEP METHOD)
        alpha = SOC_0
        stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
        stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant

        # IF no initial state is supplied to the "step" method, treat step as initial step
        xn_old = [(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ^ 2)); 0; 0] # stoi_n100 should be changed if the initial soc is not equal to 50 %
        xp_old = [(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ^ 2)); 0; 0]  # initial positive electrode ion concentration
        xe_old = [0; 0]

        Sepsi_p = [0; 0; 0]
        Sepsi_n = [0; 0; 0]
        Sdsp_p = [0; 0; 0; 0]
        Sdsn_n = [0; 0; 0; 0]

    end

    done_flag = false

    A_dp = [1. 1. 0.; 0. 1. 1.; 0. -0.003465 0.811]
    B_dp = [0.; 0.; -1.]
    C_dp = [7.1823213e-08 8.705844e-06 0.0001450974]

    A_dn = [1. 1. 0.; 0. 1. 1.; 0. -0.0005270265 0.92629]
    B_dn = [0.; 0.; -1.]
    C_dn = [9.1035395451e-09 2.82938292e-06 0.0001209138]

    Ae_dp = [0.964820774248931 0.; 0. 0.964820774248931]
    Be_dp = [6.2185e-06; 6.2185e-06]
    Ce_dp = [-39977.1776789832 0.; 0. 39977.1776789832]

    Sepsi_A_dp = [1. 1. 0.; 0. 1. 1.; 0. -0.003465 0.811]
    Sepsi_B_dp = [0.; 0.; 1.]
    Sepsi_C_dp = [0.00143646294319442 0.174116720387202 2.90194533978671]

    Sepsi_A_dn = [1. 1. 0.; .0 1. 1.; 0. -0.0005270265 0.92629]
    Sepsi_B_dn = [0.; 0.; 1.]

    I = I_input
    Jn = I / Vn
    Jp = -I / Vp

    if Jn == 0

        I = .00000001
        Jn = I / Vn
        Jp = -I / Vp

    end


    # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
    yn_new = C_dn * xn_old
    yp_new = C_dp * xp_old
    yep_new = Ce_dp * xe_old

    # println(yn_new[1])
    # println(yp_new[1])

    # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
    xn_new = A_dn * xn_old + B_dn * Jn
    xp_new = A_dp * xp_old + B_dp * Jp
    xe_new = Ae_dp * xe_old + Be_dp * I

    # Electrolyte Dynamics
    vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * log((1000 + yep_new[1] / (1000 + yep_new[2])) / F)))

    # Compute "Exchange Current Density" per Electrode (Pos & Neg)
    i_0n = kn * F * (cen * yn_new[1] * (cs_max_n - yn_new[1])) ^ .5
    i_0p = kp * F * (cep * yp_new[1] * (cs_max_p - yp_new[1])) ^ .5

    # Kappa (pos & Neg)
    k_n = Jn / (2 * as_n * i_0n)
    k_p = Jp / (2 * as_p * i_0p)

    # Compute Electrode "Overpotentials"
    eta_n = (R * T * log(k_n + (k_n ^ 2 + 1) ^ 0.5)) / (F * 0.5)
    eta_p = (R * T * log(k_p + (k_p ^ 2 + 1) ^ 0.5)) / (F * 0.5)

    # Record Stoich Ratio (SOC can be computed from this)
    theta_n = yn_new[1] / cs_max_n
    theta_p = yp_new[1] / cs_max_p

    theta = [theta_n, theta_p]  # Stoichiometry Ratio Coefficent

    SOC_n = ((theta_n - stoi_n0) / (stoi_n100 - stoi_n0))
    SOC_p = ((theta_p - stoi_p0) / (stoi_p100 - stoi_p0))

    soc_new = [SOC_n, SOC_p]

    U_n = 0.194 + 1.5 * exp(-120.0 * theta_n) + 0.0351 * tanh((theta_n - 0.286) / 0.083) - 0.0045 * tanh(
        (theta_n - 0.849) / 0.119) - 0.035 * tanh((theta_n - 0.9233) / 0.05) - 0.0147 * tanh(
        (theta_n - 0.5) / 0.034) - 0.102 * tanh((theta_n - 0.194) / 0.142) - 0.022 * tanh(
        (theta_n - 0.9) / 0.0164) - 0.011 * tanh((theta_n - 0.124) / 0.0226) + 0.0155 * tanh(
        (theta_n - 0.105) / 0.029)

    U_p = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta_p) + 2.1581 * tanh(52.294 - 50.294 * theta_p) - 0.14169 * tanh(11.0923 - 19.8543 * theta_p) + 0.2051 * tanh(1.4684 - 5.4888 * theta_p) + 0.2531 * tanh((-theta_p + 0.56478) / 0.1316) -0.02167 * tanh((theta_p - 0.525) / 0.006)

    docv_dCse_n = -1.5 * (120.0 / cs_max_n) * exp(-120.0 * theta_n) + (0.0351 / (0.083 * cs_max_n)) * ((cosh((theta_n - 0.286) / 0.083)) ^ (-2)) - (0.0045 / (cs_max_n * 0.119)) * ((cosh((theta_n - 0.849) / 0.119)) ^ (-2)) - (0.035 / (cs_max_n * 0.05)) * ((cosh((theta_n - 0.9233) / 0.05)) ^ (-2)) - (0.0147 / (cs_max_n * 0.034)) * ((cosh((theta_n - 0.5) / 0.034)) ^ (-2)) - (0.102 / (cs_max_n * 0.142)) * ((cosh((theta_n - 0.194) / 0.142)) ^ (-2)) - (0.022 / (cs_max_n * 0.0164)) * ((cosh((theta_n - 0.9) / 0.0164)) ^ (-2)) - (0.011 / (cs_max_n * 0.0226)) * ((cosh((theta_n - 0.124) / 0.0226)) ^ (-2)) + (0.0155 / (cs_max_n * 0.029)) * ((cosh((theta_n - 0.105) / 0.029)) ^ (-2))

    docv_dCse_p = 0.07645 * (-54.4806 / cs_max_p) * ((1.0 / cosh(30.834 - 54.4806 * theta_p)) ^ 2) + 2.1581 * (-50.294 / cs_max_p) * ((cosh(52.294 - 50.294 * theta_p)) ^ (-2)) + 0.14169 * (19.854 / cs_max_p) * ((cosh(11.0923 - 19.8543 * theta_p)) ^ (-2)) - 0.2051 * (5.4888 / cs_max_p) * ((cosh(1.4684 - 5.4888 * theta_p)) ^ (-2)) - 0.2531 / 0.1316 / cs_max_p * ((cosh((-theta_p + 0.56478) / 0.1316)) ^ (-2)) - 0.02167 / 0.006 /cs_max_p * ((cosh((theta_p - 0.525) / 0.006)) ^ (-2))

    theta_p = theta_p * cs_max_p
    theta_n = theta_n * cs_max_n

    # state space Output Eqn. realization for epsilon_s (Neg & Pos)
    out_Sepsi_p = Sepsi_C_dp * Sepsi_p

    out_Sepsi_p = out_Sepsi_p[1]
    # println("out_Sepsi_p: $(out_Sepsi_p)")
    # state space realization for epsilon_s (Neg & Pos)
    Sepsi_p_new = Sepsi_A_dp * Sepsi_p + Sepsi_B_dp * I  # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
    # println("Sepsi_p_new: $(Sepsi_p_new)")
    Sepsi_n_new = Sepsi_A_dn * Sepsi_n + Sepsi_B_dn * I
    # println("Sepsi_n_new: $(Sepsi_n_new)")


    # rho1p_1 = -sign(I) * (-3 * R * T) / (0.5 * F * Rp * as_p) * ((1 + 1 / k_p ^ 2) ^ (-0.5))
    rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ^ 2 + 1) ^ 0.5)) * (1 + k_p / ((k_p ^ 2 + 1) ^ 0.5)) * (-3 * Jp / (2 * as_p ^ 2 * i_0p * Rp))

    rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (cep * theta_p * (cs_max_p - theta_p)) * (1 + (1 / (k_p + .00000001) ^ 2)) ^ (-0.5)

    # rho1n_1 = sign(I) * (-3 * R * T) / (0.5 * F * Rn * as_n) * ((1 + 1 / k_n ^ 2) ^ (-0.5))
    rho1n = R * T / (0.5 * F) * (1 / (k_n + (k_n ^ 2 + 1) ^ 0.5)) * (1 + k_n / ((k_n ^ 2 + 1) ^ 0.5)) * (-3 * Jn / (2 * as_n ^ 2 * i_0n * Rn))

    rho2n = (-R * T) / (2 * 0.5 * F) * (cen * cs_max_n - 2 * cen * theta_n) / (cen * theta_n * (cs_max_n - theta_n)) * (1 + 1 / (k_n + .00000001) ^ 2) ^ (-0.5)

    # sensitivity of epsilon_sp epsilon_sn
    sen_out_spsi_p = (rho1p + (rho2p + docv_dCse_p) * -out_Sepsi_p)

    out_deta_p_desp = rho1p + rho2p * (-1) * out_Sepsi_p
    out_semi_linear_p = docv_dCse_p * out_Sepsi_p

    dV_dEpsi_sp = sen_out_spsi_p

    # Surface Concentration Sensitivity for Epsilon (pos & neg)
    dCse_dEpsi_sp = -1. * out_Sepsi_p * epsi_n

    docv_dCse = [docv_dCse_n, docv_dCse_p]

    V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
    R_film = -Rf * I / (Ar_n * Ln * as_n)

    return xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag
end

end

# # @profile
# def main():
#
#     num_episodes = 1
#     soc_list = []
#
#     xn = nothing
#     xp = nothing
#     xe = nothing
#     Sepsi_p = nothing
#     Sepsi_n = nothing
#
#     min_SOC = 0
#     max_SOC = 1.1
#     min_volt = 2.25
#     max_volt = 4.4
#
#
#     # init_time = time.time_ns()
#     for num_eps = 1:length(num_episodes)
#         for t = 1:length(1800)
#
#             initial_step = t ==0 ? true : false
#
#             [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] =\
#                 SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input=-25.7, init_flag=initial_step)
#
#             V_term = V_term.item()
#             soc_val = soc_new[0]
#             xn = xn_new
#             xp = xp_new
#             xe = xe_new
#             Sepsi_p = Sepsi_p_new
#             Sepsi_n = Sepsi_n_new
#
#             soc_list.append(soc_val.item())
#
#             if soc_val <= min_SOC or soc_val >= max_SOC or V_term <= min_volt or V_term >= max_volt:
#
#                 break
#             end
#         end
#     end
#
#
#     plt.plot(soc_list)
#     plt.show()
