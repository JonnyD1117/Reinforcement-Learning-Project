%% Jonathan SPMe Model Code 

clear all 
clc
states = 0;

I_input = -25.7;
full_sim = 0;
init_flag = 1; 

xn = 0; 
xp = 0; 
xe = 0; 
Sepsi_p = 0; 
Sepsi_n = 0; 
Sdsp_p = 0; 
Sdsn_n = 0; 

soc_list = .5;

for t = 0:1:1800
    
    if t ==0
        init_flag =1; 
        
    else
        init_flag = 0; 
    end 
    
    [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, Sdsp_p_new, Sdsn_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] = SPMe_step( xn, xp, xe, Sepsi_p, Sepsi_n, Sdsp_p, Sdsn_n, I_input, full_sim, init_flag);

    
    
    soc_list = [soc_list, soc_new(1)];
    
    
     xn =xn_new;
     xp =xp_new;
     xe =xe_new;
     Sepsi_p =Sepsi_p_new;
     Sepsi_n =Sepsi_n_new;
     Sdsp_p =Sdsp_p_new;
     Sdsn_n =Sdsn_n_new;
     
end 


figure()
plot(soc_list)
title("SOC")
xlabel("Time")
ylabel("State of Charge")





%% Function Definition(s)

function [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, Sdsp_p_new, Sdsn_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] = SPMe_step( xn, xp, xe, Sepsi_p, Sepsi_n, Sdsp_p, Sdsn_n, I_input, full_sim, init_flag)

    SOC_0 = .5;

    epsilon_sn = 0.6;  % % # average negative active volume fraction
    epsilon_sp = 0.50;  % % # average positive active volume fraction
    epsilon_e_n = 0.3;  % % # Liquid [electrolyte] volume fraction (pos & neg)
    epsilon_e_p = 0.3;

    F = 96487;  % # Faraday constant
    Rn = 10e-6;  % # Active particle radius (pose & neg)
    Rp = 10e-6;

    R = 8.314;  % # Universal gas constant
    T = 298.15;  % # Ambient Temp. (kelvin)

    Ar_n = 1;  % # Current collector area (anode & cathode)
    Ar_p = 1;

    Ln = 100e-6;  % # Electrode thickness (pos & neg)
    Lp = 100e-6;
    Lsep = 25e-6;  % # Separator Thickness
    Lc = Ln + Lp + Lsep;  % # Total Cell Thickness

    Ds_n = 3.9e-14;  % # Solid phase diffusion coefficient (pos & neg)
    Ds_p = 1e-13;
    De = 2.7877e-10;  % # Electrolyte Diffusion Coefficient
    De_p = De;
    De_n = De;

    kn = 1e-5 / F;  % # Rate constant of exchange current density (Reaction Rate) [Pos & neg]
    kp = 3e-7 / F;

    % # Stoichiometric Coef. used for "interpolating SOC value based on OCV Calcs. at 0.0069% and 0.8228%
    stoi_n0 = 0.0069;  % # Stoich. Coef. for Negative Electrode
    stoi_n100 = 0.6760;

    stoi_p0 = 0.8228;  % # Stoich. Coef for Positive Electrode
    stoi_p100 = 0.442;

    SOC = 1;  % # SOC can change from 0 to 1

    cs_max_n = (3.6e3 * 372 * 1800) / F;  % # 0.035~0.870=1.0690e+03~ 2.6572e+04
    cs_max_p = (3.6e3 * 274 * 5010) / F;  % # Positive electrode  maximum solid-phase concentration 0.872~0.278=  4.3182e+04~1.3767e+04

    Rf = 1e-3;  % #
    as_n = 3 * epsilon_sn / Rn;  % # Active surface area per electrode unit volume (Pos & Neg)
    as_p = 3 * epsilon_sp / Rp;

    Vn = Ar_n * Ln;  % # Electrode volume (Pos & Neg)
    Vp = Ar_p * Lp;

    t_plus = 0.4;

    cep = 1000;  % # Electrolyte Concentration (Assumed Constant?) [Pos & Neg]
    cen = 1000;

    % # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
    rfa_n = 1 / (F * as_n * Rn ^ 5);
    rfa_p = 1 / (F * as_p * Rp ^ 5);

    epsi_sep = 1;
    epsi_e = 0.3;
    epsi_n = epsi_e;
    gamak = (1 - t_plus) / (F * Ar_n);

    kappa = 1.1046;
    kappa_eff = kappa * (epsi_e ^ 1.5);
    kappa_eff_sep = kappa * (epsi_sep ^ 1.5);

    % #% ###################################################################################################################
    % ####################################################################################################################
    % # Simulation Settings
    Ts = 1;

    % # Default Input "Current" Settings
    default_current = 25.67;  % # Base Current Draw

    if (init_flag == 1)

        % # Initialize the "battery" and 'sensitivity' states (FOR STEP METHOD)
        alpha = SOC_0;
        stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0;  % # Negative Electrode Interpolant
        stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha;  % # Positive Electrode Interpolant

        % # IF no initial state is supplied to the "step" method, treat step as initial step
        xn_old = [(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ^ 2)); 0 ; 0];  % # stoi_n100 should be changed if the initial soc is not equal to 50 %
        xp_old = [(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ^ 2)); 0 ; 0]; % # initial positive electrode ion concentration
        xe_old = [0 ; 0];

        Sepsi_p_old = [0; 0; 0];
        Sepsi_n_old = [0; 0; 0];
        Sdsp_p_old = [0; 0; 0; 0];
        Sdsn_n_old = [0; 0; 0; 0];


    else

%         if (states == 0)
%             % raise Exception("System States are of type NONE!")
%             disp("System States are of type NONE!");
%         end

        % # Unpack Initial battery state variables from dict for use in state space computation
        xn_old = xn;
        xp_old = xp;
        xe_old = xe;
        
        Sepsi_p_old = Sepsi_p;
        Sepsi_n_old = Sepsi_n;
        Sdsp_p_old = Sdsp_p;
        Sdsn_n_old = Sdsn_n;
        
        

        
    end 
    
    % # Set DONE flag to false: NOTE - done flag indicates whether model encountered invalid state. This flag is exposed
    % # to the user via the output and is used to allow the "step" method to terminate higher level functionality/simulations.
    done_flag = 0;

    Ap = [0, 1, 0; 0, 0, 1; 0, -(3465 * (Ds_p ^ 2) / Rp ^ 4), - (189 * Ds_p / Rp ^ 2)];
    Bp = [0; 0; -1];
    Cp = rfa_p * [10395 * Ds_p ^ 2, 1260 * Ds_p * Rp ^ 2, 21 * Rp ^ 4];
    Dp = [0];

    [n_pos, m_pos] = size(Ap);
    A_dp = eye(n_pos) + Ap * Ts;
    B_dp = Bp * Ts;
    C_dp = Cp;
    D_dp = Dp;

    % # Negative electrode three-state state space model for the particle
    An = [0, 1, 0; 0, 0, 1; 0, - (3465 * (Ds_n ^ 2) / Rn ^ 4),- (189 * Ds_n / Rn ^ 2)];
    Bn = [0; 0; -1];
    Cn = rfa_n * [10395 * Ds_n ^ 2, 1260 * Ds_n * Rn ^ 2, 21 * Rn ^ 4];
    Dn = [0];
    
    % # Negative electrode SS Discretized
    [n_neg, m_neg] = size(An);
    A_dn = eye(n_neg) + An * Ts;
    B_dn = Bn * Ts;
    C_dn = Cn;
    D_dn = Dn;

    % # electrolyte  concentration (boundary)
    a_p0 = -(epsi_n ^ (3 / 2) + 4 * epsi_sep ^ (3 / 2)) / (80000 * De_p * epsi_n ^ (3 / 2) * epsi_sep ^ (3 / 2));
    b_p0 = (epsi_n ^ 2 * epsi_sep + 24 * epsi_n ^ 3 + 320 * epsi_sep ^ 3 + 160 * epsi_n ^ (3 / 2) * epsi_sep ^ (3 / 2)) / (19200000000 * (4 * De_p * epsi_n ^ (1 / 2) * epsi_sep ^ 3 + De_p * epsi_n ^ 2 * epsi_sep ^ (3 / 2)));

    a_n0 = (epsi_n ^ (3 / 2) + 4 * epsi_sep ^ (3 / 2)) / (80000 * De * epsi_n ^ (3 / 2) * epsi_sep ^ (3 / 2));
    b_n0 = (epsi_n ^ 2 * epsi_sep + 24 * epsi_n ^ 3 + 320 * epsi_sep ^ 3 + 160 * epsi_n ^ (3 / 2) * epsi_sep ^ (3 / 2)) / (19200000000 * (4 * De_n * epsi_n ^ (1 / 2) * epsi_sep ^ 3 + De_n * epsi_n ^ 2 * epsi_sep ^ (3 / 2)));

    Aep = [-1 / b_p0, 0; 0, -1 / b_n0];
    Bep = gamak * [1; 1];
    Cep = [a_p0 / b_p0, 0; 0, a_n0 / b_n0];
    Dep = [0];

    [n_elec, m] = size(Aep);
    Ae_dp = eye(n_elec) + Aep * Ts;
    Be_dp = Bep * Ts;
    Ce_dp = Cep;
    De_dp = Dep;

    coefp = 3 / (F * Rp ^ 6 * as_p ^ 2 * Ar_p * Lp);
    Sepsi_A_p = [0, 1, 0; 0, 0, 1; 0, -(3465 * Ds_p ^ 2) / Rp ^ 4, -(189 * Ds_p) / Rp ^ 2];
    Sepsi_B_p = [0; 0; 1];
    Sepsi_C_p = coefp * [10395 * Ds_p ^ 2, 1260 * Ds_p * Rp ^ 2, 21 * Rp ^ 4];
    Sepsi_D_p = [0];

    [n, m] = size(Sepsi_A_p);
    Sepsi_A_dp = eye(n) + Sepsi_A_p * Ts;
    Sepsi_B_dp = Sepsi_B_p * Ts;
    Sepsi_C_dp = Sepsi_C_p;
    Sepsi_D_dp = Sepsi_D_p;

    % # sensitivity realization in time domain for epsilon_sn from third order pade(you can refer to my slides)
    coefn = 3 / (F * Rn ^ 6 * as_n ^ 2 * Ar_n * Ln);

    Sepsi_A_n = [0, 1, 0; 0, 0, 1; 0, -(3465 * Ds_n ^ 2) / Rn ^ 4, -(189 * Ds_n) / Rn ^ 2];
    Sepsi_B_n = [0; 0; 1];
    Sepsi_C_n = coefn * [10395 * Ds_n ^ 2, 1260 * Ds_n * Rn ^ 2, 21 * Rn ^ 4];
    Sepsi_D_n = [0];

    [n, m] = size(Sepsi_A_n);
    Sepsi_A_dn = eye(n) + Sepsi_A_n * Ts;
    Sepsi_B_dn = Sepsi_B_n * Ts;
    Sepsi_C_dn = Sepsi_C_n;
    Sepsi_D_dn = Sepsi_D_n;

    % # sensitivity realization in time domain for D_sp from third order pade
    coefDsp = (63 * Rp) / (F * as_p * Ar_p * Lp * Rp ^ 8);

    Sdsp_A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; -(12006225 * Ds_p ^ 4) / Rp ^ 8, -1309770 * Ds_p ^ 3 / Rp ^ 6, -42651 * Ds_p ^ 2 / Rp ^ 4, -378 * Ds_p / Rp ^ 2];
    Sdsp_B = [0; 0; 0; 1];
    Sdsp_C = coefDsp * [38115 * Ds_p ^ 2, 1980 * Ds_p * Rp ^ 2, 43 * Rp ^ 4, 0];
    Sdsp_D = [0];

    [n, m] = size(Sdsp_A);
    Sdsp_A_dp = eye(n) + Sdsp_A * Ts;
    Sdsp_B_dp = Sdsp_B * Ts;
    Sdsp_C_dp = Sdsp_C;
    Sdsp_D_dp = Sdsp_D;

    % # sensitivity realization in time domain for D_sn from third order pade
    coefDsn = (63 * Rn) / (F * as_n * Ar_n * Ln * Rn ^ 8);

    Sdsn_A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; -(12006225 * Ds_n ^ 4) / Rn ^ 8, -1309770 * Ds_n ^ 3 / Rn ^ 6, -42651 * Ds_n ^ 2 / Rn ^ 4, -378 * Ds_n / Rn ^ 2];
    Sdsn_B = [0; 0; 0; 1];
    Sdsn_C = coefDsn * [38115 * Ds_n ^ 2, 1980 * Ds_n * Rn ^ 2, 43 * Rn ^ 4, 0];
    Sdsn_D = [0];

    [n, m] = size(Sdsn_A);
    Sdsn_A_dn = eye(n) + Sdsn_A * Ts;
    Sdsn_B_dn = Sdsn_B * Ts;
    Sdsn_C_dn = Sdsn_C;
    Sdsn_D_dn = Sdsn_D;

    % # If FULL SIM is set True: Shortciruit SIM "I" & "SOC" values into step model (Does not Check for None inputs or default values)
    if full_sim == 1
        I = I_input;

        if I == 0
            % # I = .000000001
            I = 0.0;
        end
    else
        % # Initialize Input Current
        if I_input == []
            I = default_current;  % # If no input signal is provided use CC @ default input value
        else
            I = I_input;
        end
    end

        % # Molar Current Flux Density (Assumed UNIFORM for SPM)
    Jn = I / Vn;
    Jp = -I / Vp;

    if Jn == 0
        % # print("Molar Current Density (Jn) is equal to zero. This causes 'division by zero' later")
        % # print("I", I)

        I = .00000001;

        Jn = I / Vn;
        Jp = -I / Vp;
    end

    % # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
    yn_new = C_dn * xn_old + D_dn * 0;
    yp_new = C_dp * xp_old + D_dp * 0;
    yep_new = Ce_dp * xe_old + De_dp * 0;

    % # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
    xn_new = A_dn * xn_old + B_dn * Jn;
    xp_new = A_dp * xp_old + B_dp * Jp;
    xe_new = Ae_dp * xe_old + Be_dp * I;

    % # Electrolyte Dynamics
    vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * log((1000 + yep_new(1)) / (1000 + yep_new(2)))) / F);  % # yep(1, k) = positive boundary;

    % # R_e = -I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep)
    % # V_con = (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F
    % # phi_n = 0
    % # phi_p = phi_n + vel

    % # Compute "Exchange Current Density" per Electrode (Pos & Neg)
    i_0n = kn * F * (cen * yn_new * (cs_max_n - yn_new)) ^ .5;
    i_0p = kp * F * (cep * yp_new * (cs_max_p - yp_new)) ^ .5;

    % # Kappa (pos & Neg)
    k_n = Jn / (2 * as_n * i_0n);
    k_p = Jp / (2 * as_p * i_0p);

    % # Compute Electrode "Overpotentials"
    eta_n = (R * T * log(k_n + (k_n ^ 2 + 1) ^ 0.5)) / (F * 0.5);
    eta_p = (R * T * log(k_p + (k_p ^ 2 + 1) ^ 0.5)) / (F * 0.5);

    % # Record Stoich Ratio (SOC can be computed from this)
    theta_n = yn_new / cs_max_n;
    theta_p = yp_new / cs_max_p;

    theta = [theta_n, theta_p];  % # Stoichiometry Ratio Coefficent

    SOC_n = ((theta_n - stoi_n0) / (stoi_n100 - stoi_n0));
    SOC_p = ((theta_p - stoi_p0) / (stoi_p100 - stoi_p0));

    soc_new = [SOC_n, SOC_p];

    U_n = 0.194 + 1.5 * exp(-120.0 * theta_n) + 0.0351 * tanh((theta_n - 0.286) / 0.083) - 0.0045 * tanh((theta_n - 0.849) / 0.119) - 0.035 * tanh((theta_n - 0.9233) / 0.05) - 0.0147 * tanh((theta_n - 0.5) / 0.034) - 0.102 * tanh((theta_n - 0.194) / 0.142) - 0.022 * tanh((theta_n - 0.9) / 0.0164) - 0.011 * tanh((theta_n - 0.124) / 0.0226) + 0.0155 * tanh((theta_n - 0.105) / 0.029);
    U_p = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta_p) + 2.1581 * tanh(52.294 - 50.294 * theta_p) - 0.14169 * tanh(11.0923 - 19.8543 * theta_p) + 0.2051 * tanh(1.4684 - 5.4888 * theta_p) + 0.2531 * tanh((-theta_p + 0.56478) / 0.1316) - 0.02167 * tanh((theta_p - 0.525) / 0.006);

    docv_dCse_n = -1.5 * (120.0 / cs_max_n) * exp(-120.0 * theta_n) + (0.0351 / (0.083 * cs_max_n)) * ((cosh((theta_n - 0.286) / 0.083)) ^ (-2)) - (0.0045 / (cs_max_n * 0.119)) * ((cosh((theta_n - 0.849) / 0.119)) ^ (-2)) - (0.035 / (cs_max_n * 0.05)) * ((cosh((theta_n - 0.9233) / 0.05)) ^ (-2)) - (0.0147 / (cs_max_n * 0.034)) * ((cosh((theta_n - 0.5) / 0.034)) ^ (-2)) - (0.102 / (cs_max_n * 0.142)) * ((cosh((theta_n - 0.194) / 0.142)) ^ (-2)) - (0.022 / (cs_max_n * 0.0164)) * ((cosh((theta_n - 0.9) / 0.0164)) ^ (-2)) - (0.011 / (cs_max_n * 0.0226)) * ((cosh((theta_n - 0.124) / 0.0226)) ^ (-2)) + (0.0155 / (cs_max_n * 0.029)) * ((cosh((theta_n - 0.105) / 0.029)) ^ (-2));
    docv_dCse_p = 0.07645 * (-54.4806 / cs_max_p) * ((1.0 / cosh(30.834 - 54.4806 * theta_p)) ^ 2) + 2.1581 * (-50.294 / cs_max_p) * ((cosh(52.294 - 50.294 * theta_p)) ^ (-2)) + 0.14169 * (19.854 / cs_max_p) * ((cosh(11.0923 - 19.8543 * theta_p)) ^ (-2)) - 0.2051 * (5.4888 / cs_max_p) * ((cosh(1.4684 - 5.4888 * theta_p)) ^ (-2)) - 0.2531 / 0.1316 / cs_max_p * ((cosh((-theta_p + 0.56478) / 0.1316)) ^ (-2)) - 0.02167 / 0.006 /cs_max_p * ((cosh((theta_p - 0.525) / 0.006)) ^ (-2));

    theta_p = theta_p * cs_max_p;
    theta_n = theta_n * cs_max_n;

    % # state space Output Eqn. realization for epsilon_s (Neg & Pos)
    out_Sepsi_p = Sepsi_C_dp * Sepsi_p;
    out_Sepsi_n = Sepsi_C_dn * Sepsi_n;

    % # state space Output Eqn. realization for D_s (neg and Pos)
    out_Sdsp_p = Sdsp_C_dp * Sdsp_p;
    out_Sdsn_n = Sdsn_C_dn * Sdsn_n;

    % # state space realization for epsilon_s (Neg & Pos)
    Sepsi_p_new = Sepsi_A_dp * Sepsi_p + Sepsi_B_dp * I;  % # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
    Sepsi_n_new = Sepsi_A_dn * Sepsi_n + Sepsi_B_dn * I;

    % # state space realization for D_s (neg and Pos)
    Sdsp_p_new = Sdsp_A_dp * Sdsp_p + Sdsp_B_dp * I;
    Sdsn_n_new = Sdsn_A_dn * Sdsn_n + Sdsn_B_dn * I;

    % # rho1p_1 = -np.sign(I) * (-3 * R * T) / (0.5 * F * Rp * as_p) * ((1 + 1 / k_p ^ 2) ^ (-0.5))
    rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ^ 2 + 1) ^ 0.5)) * (1 + k_p / ((k_p ^ 2 + 1) ^ 0.5)) * (-3 * Jp / (2 * as_p ^ 2 * i_0p * Rp));

    rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (cep * theta_p * (cs_max_p - theta_p)) * (1 + (1 / (k_p + .00000001) ^ 2)) ^ (-0.5);

    % # rho1n_1 = np.sign(I) * (-3 * R * T) / (0.5 * F * Rn * as_n) * ((1 + 1 / k_n ^ 2) ^ (-0.5))
    rho1n = R * T / (0.5 * F) * (1 / (k_n + (k_n ^ 2 + 1) ^ 0.5)) * (1 + k_n / ((k_n ^ 2 + 1) ^ 0.5)) * (-3 * Jn / (2 * as_n ^ 2 * i_0n * Rn));

    rho2n = (-R * T) / (2 * 0.5 * F) * (cen * cs_max_n - 2 * cen * theta_n) / (cen * theta_n * (cs_max_n - theta_n)) * (1 + 1 / (k_n + .00000001) ^ 2) ^ (-0.5);

    % # sensitivity of epsilon_sp epsilon_sn
    sen_out_spsi_p = (rho1p + (rho2p + docv_dCse_p) * -out_Sepsi_p);
    sen_out_spsi_n = (rho1n + (rho2n + docv_dCse_n) * out_Sepsi_n);

    out_deta_p_desp = rho1p + rho2p * (-1) * out_Sepsi_p;
    out_deta_n_desn = rho1n + rho2n * out_Sepsi_n;

    out_semi_linear_p = docv_dCse_p * out_Sepsi_p;
    out_semi_linear_n = docv_dCse_n * out_Sepsi_n;

    % # sensitivity of Dsp Dsn
    sen_out_ds_p = ((rho2p + docv_dCse_p) * (-1 * out_Sdsp_p)) * Ds_p;
    sen_out_ds_n = ((rho2n + docv_dCse_n) * out_Sdsn_n) * Ds_n;

    dV_dDsp = sen_out_ds_p;
    dV_dDsn = sen_out_ds_n;

    dV_dEpsi_sn = sen_out_spsi_n;
    dV_dEpsi_sp = sen_out_spsi_p;

    dCse_dDsp = -1 * out_Sdsp_p * Ds_p;
    dCse_dDsn = out_Sdsn_n * Ds_n;

    % # Surface Concentration Sensitivity for Epsilon (pos & neg)
    dCse_dEpsi_sp = -1. * out_Sepsi_p * epsi_n;
    dCse_dEpsi_sn = out_Sepsi_n * epsi_n;  % # Espi_N and Epsi_p have the same value, Epsi_p currently not defined


    docv_dCse = [docv_dCse_n, docv_dCse_p];

    V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n);  % # terminal voltage
    R_film = -Rf * I / (Ar_n * Ln * as_n);

    
end 
   



