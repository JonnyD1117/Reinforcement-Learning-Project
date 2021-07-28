%% Jonathan SPMe Optimized Q-Learning 

rng(0);

clear 
close all
clc

% Training Duration Parameters
num_episodes =1000 ;
episode_duration = 1800;

% Initialize Q-Learning Table
num_q_states = 1000;
num_q_actions = 101;
%     num_q_states = 250;
%     num_q_actions = 11;

% Discretization Parameters
max_state_val = 1;
min_state_val = 0;
% max_action_val = 25.7;
% min_action_val = -25.7;
max_action_val = 25.7*3;
min_action_val = -25.7*3;

[action_values, action_index, action_dict] = Discretization_Dict([min_action_val, max_action_val], num_q_actions);
[soc_state_values, soc_state__index, soc_state_dict] = Discretization_Dict([min_state_val, max_state_val], num_q_states);

% Q-Learning Parameters
Q = zeros(num_q_states+1, num_q_actions+1);
alpha = .1;
epsilon = .05;
gamma = .98;

% SPMe Initialization Parameters
SOC_0 = .5;
% [soc_state_value, soc_state_index] = Discretize_Value(SOC_0, soc_state_values);

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

time_list = [];

% spme_params = load("C:\Users\Indy-Windows\Desktop\SPMe_opt3_const_vars.mat");
spme_params = 0; 

voltage_list = [] ; 
soc_list = []; 
current_list = [] ; 


    for eps = 1:1:num_episodes
    %     disp(eps)

        if mod(eps, 1000) == 0
            disp(eps)
    %         epsilon = epsilon*.75;
        end 
        
%         if eps <= 100000
%             epsilon = epsilon*.75;
%         else
%             epsilon = .05;
%         end 

        [state_value, state_index] = Discretize_Value(SOC_0, soc_state_values);

        for step = 1:1:episode_duration
    %         disp(step)

            %# Select Action According to Epsilon Greedy Policy
            action_index = epsilon_greedy_policy(state_index, Q, [num_q_states, num_q_actions], epsilon);

            %# Given the action Index find the corresponding action value                                
        if step ==1
            init_flag =1; 
            xn = 1.0e+11*[9.3705; 0; 0];
            xp = 1.0e+11*[4.5097; 0; 0];
            xe = [0;0];
            Sepsi_p = [0; 0; 0];
            Sepsi_n = [0; 0; 0];

        end 

            I_input = action_values(action_index);

            [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new,dV_dEpsi_sp,...
            soc_new, V_term, theta, docv_dCse, done_flag] = SPMe_step( xn, xp, xe, Sepsi_p, Sepsi_n, I_input, full_sim, init_flag, spme_params);

             soc = (soc_new(1)+soc_new(2))/2;

             xn =xn_new;
             xp =xp_new;
             xe =xe_new;
             Sepsi_p =Sepsi_p_new;
             Sepsi_n =Sepsi_n_new;

            %# Discretize the Resulting SOC
            [new_state_value, new_state_index] = Discretize_Value(soc, soc_state_values);

            %# Compute Reward Function
            R = dV_dEpsi_sp(1)^2;

            %# Update Q-Function
            [max_Q, max_Q_ind] = max(Q(new_state_index, :));

            Q(state_index,action_index) = Q(state_index,action_index) + alpha*(R + gamma*max_Q - Q(state_index, action_index));

            state_value = new_state_value;
            state_index = new_state_index;

            if( soc >= 1 || V_term >= 4.4)

                break;
             end 
        end 
    end


% figure()
% plot(episode_times)
% title("Total Episode Times")


% figure()
% plot(soc_list)
% 
% figure()
% plot(voltage_list)
% figure()
% plot(current_list)

% 	Q;
% 
%     final_time = toc(t_init);
% 
    [action_list, soc_val_list] = q_learning_policy(Q, num_q_states, num_q_actions, [min_state_val,max_state_val ], [min_action_val, max_action_val] );

    figure()
    plot(action_list)
    title("Input Current")
        %
    figure()
    plot(soc_val_list)
    title("SOC Output")





%% Function Definition(s)
% CHECKED
function [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] = SPMe_step( xn, xp, xe, Sepsi_p, Sepsi_n, I_input, full_sim, ~, ~)

    done_flag =0;
    F = 96487;  % # Faraday constant
    Rp = 10e-6;
    R = 8.314;  % # Universal gas constant
    T = 298.15;  % # Ambient Temp. (kelvin)
    Ar_n = 1;  % # Current collector area (anode & cathode)
    Ln = 100e-6;  % # Electrode thickness (pos & neg)
    Lp = 100e-6;
    Lsep = 25e-6;  % # Separator Thickness
    kn = 1.0364e-10; %1e-5 / F;  % # Rate constant of exchange current density (Reaction Rate) [Pos & neg]
    kp = 3.1092e-12; %3e-7 / F;
    % # Stoichiometric Coef. used for "interpolating SOC value based on OCV Calcs. at 0.0069% and 0.8228%
    stoi_n0 = 0.0069;  % # Stoich. Coef. for Negative Electrode
    stoi_n100 = 0.6760;
    stoi_p0 = 0.8228;  % # Stoich. Coef for Positive Electrode
    stoi_p100 = 0.442;
    cs_max_n = 2.4983e+04;% (3.6e3 * 372 * 1800) / F;  % # 0.035~0.870=1.0690e+03~ 2.6572e+04
    cs_max_p = 5.1218e+04;% (3.6e3 * 274 * 5010) / F;  % # Positive electrode  maximum solid-phase concentration 0.872~0.278=  4.3182e+04~1.3767e+04
    Rf = 1e-3;  % #
    as_n = 1.8000e+05;% 3 * epsilon_sn / Rn;  % # Active surface area per electrode unit volume (Pos & Neg)
    as_p = 150000; % 3 * epsilon_sp / Rp;
    Vn =1.0000e-04; %  Ar_n * Ln;  % # Electrode volume (Pos & Neg)
    Vp = 1.0000e-04; % Ar_p * Lp;
    t_plus = 0.4;
    cep = 1000;  % # Electrolyte Concentration (Assumed Constant?) [Pos & Neg]
    cen = 1000;
    % # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
    kappa_eff = 0.1815; % kappa * (epsi_e ^ 1.5);
    kappa_eff_sep = 1.1046; % kappa * (epsi_sep ^ 1.5);

    A_dp = [1,1,0;0,1,1;0,-0.003465,0.811]   ;
    B_dp =   [0;0;-1]  ;
    C_dp =  [7.1823213e-08,8.705844e-06,0.0001450974]  ;

    A_dn =   [1,1,0;0,1,1;0,-0.0005270265,0.92629]  ;
    B_dn =   [0;0;-1]  ;
    C_dn =  [9.1035395451e-09,2.82938292e-06,0.0001209138]    ;

    Ae_dp =   [0.964820774248931,0;0,0.964820774248931]   ;
    Be_dp =   [6.2185e-06;6.2185e-06]   ;
    Ce_dp =    [-39977.1776789832,0;0,39977.1776789832]   ;

    Sepsi_A_dp = [1,1,0;0,1,1;0,-0.003465,0.811]    ;
    Sepsi_B_dp =   [0;0;1]   ;
    Sepsi_C_dp =  [0.00143646294319442,0.174116720387202,2.90194533978671]   ;

    Sepsi_A_dn = [1,1,0;0,1,1;0,-0.0005270265,0.92629]   ;
    Sepsi_B_dn =  [0;0;1]   ;


        xn_old = xn;
        xp_old = xp;
        xe_old = xe;

    if full_sim == 1
        I = I_input;
        

        if I == 0
            % # I = .000000001
            I = 0.0;
        end
    else
        % # Initialize Input Current
        
        I = I_input;
    end

        % # Molar Current Flux Density (Assumed UNIFORM for SPM)
    Jn = I / Vn;
    Jp = -I / Vp;

    if Jn == 0

        I = .00000001;

        Jn = I / Vn;
        Jp = -I / Vp;
    end

    % # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
    yn_new = C_dn * xn_old ;
    yp_new = C_dp * xp_old ;
    yep_new = Ce_dp * xe_old ;

    % # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
    xn_new = A_dn * xn_old + B_dn * Jn;
    xp_new = A_dp * xp_old + B_dp * Jp;
    xe_new = Ae_dp * xe_old + Be_dp * I;

    % # Electrolyte Dynamics
    vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * log((1000 + yep_new(1)) / (1000 + yep_new(2)))) / F);  % # yep(1, k) = positive boundary;

    % # Compute "Exchange Current Density" per Electrode (Pos & Neg)

%     i_0n = 9.99991268e-06 * ((24983000 * yn_new - cen * yn_new *yn_new)) ^ .5;
%     i_0p = 2.999973804e-07 * ((51218000 * yp_new - cep * yp_new *yp_new)) ^ .5;
    
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

    docv_dCse_n = -0.00720489933154545 * exp(-120.0 * theta_n) + 1.69271731283297e-05 * ((cosh((theta_n - 0.286) / 0.083)) ^ (-2)) - 1.51363431334988e-06 * ((cosh((theta_n - 0.849) / 0.119)) ^ (-2)) - 2.80190529560101e-05 * ((cosh((theta_n - 0.9233) / 0.05)) ^ (-2)) - 1.73058856493003e-05 * ((cosh((theta_n - 0.5) / 0.034)) ^ (-2)) - 2.87519456892659e-05 * ((cosh((theta_n - 0.194) / 0.142)) ^ (-2)) - 5.36950492188347e-05 * ((cosh((theta_n - 0.9) / 0.0164)) ^ (-2)) - 1.94822744953294e-05 * ((cosh((theta_n - 0.124) / 0.0226)) ^ (-2)) + 2.13938581683821e-05 * ((cosh((theta_n - 0.105) / 0.029)) ^ (-2));
    docv_dCse_p = -8.13198850013667e-05 * ((1.0 / cosh(30.834 - 54.4806 * theta_p)) ^ 2) -0.00211916672654145 * ((cosh(52.294 - 50.294 * theta_p)) ^ (-2)) + 5.49243090319809e-05 * ((cosh(11.0923 - 19.8543 * theta_p)) ^ (-2)) - 2.19796337225194e-05 * ((cosh(1.4684 - 5.4888 * theta_p)) ^ (-2)) - 3.75503198023206e-05 * ((cosh((-theta_p + 0.56478) / 0.1316)) ^ (-2)) - 7.05155739518659e-05 * ((cosh((theta_p - 0.525) / 0.006)) ^ (-2));


    theta_p = theta_p * cs_max_p;

    % # state space Output Eqn. realization for epsilon_s (Neg & Pos)
    out_Sepsi_p = Sepsi_C_dp * Sepsi_p;

    % # state space realization for epsilon_s (Neg & Pos)
    Sepsi_p_new = Sepsi_A_dp * Sepsi_p + Sepsi_B_dp * I;  % # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
    Sepsi_n_new = Sepsi_A_dn * Sepsi_n + Sepsi_B_dn * I;


    rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ^ 2 + 1) ^ 0.5)) * (1 + k_p / ((k_p ^ 2 + 1) ^ 0.5)) * (-3 * Jp / (2 * as_p ^ 2 * i_0p * Rp));

    rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (cep * theta_p * (cs_max_p - theta_p)) * (1 + (1 / (k_p + .00000001) ^ 2)) ^ (-0.5);

    sen_out_spsi_p = (rho1p + (rho2p + docv_dCse_p) * -out_Sepsi_p);

    dV_dEpsi_sp = sen_out_spsi_p;

    docv_dCse = [docv_dCse_n, docv_dCse_p];

    V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n);  % # terminal voltage

    
end
% CHECKED
function [discrete_values, index_values, zipped_var] = Discretization_Dict(input_range, num_disc)

    step_size = (input_range(2) - input_range(1)) / num_disc ;                    %# Compute the Average Step-Size for "num_disc" levels
    discrete_values = [input_range(1):step_size:input_range(2)];  
    index_values = [1:1:(num_disc+1)]; 
    zipped_var = containers.Map(index_values, discrete_values);
end 
% CHECKED
function [output_val, output_index ] =  Discretize_Value(input_val, input_values)
    
    [row, col] = size(input_values);    
    input_vect = input_val.*ones(row, col);    
    [~, argmin] = min(abs(input_values-input_vect));
    
    
    output_val = input_values(argmin);
    output_index = argmin;     
end
% CHECKED
function action_ind =  epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon)
    num_act = Q_dims(2);

    if rand() <= epsilon
        action_ind = randi([1 num_act+1],1,1);

    else        
        [~, action_ind] = max(Q_table(state_ind, :));
    end 
end    

function [action_list, soc_list] = q_learning_policy(Q_table, num_states, num_actions, state_range, action_range)

    % Discretization Parameters
    max_state_val = state_range(2);
    min_state_val = state_range(1);

    max_action_val = action_range(2); 
    min_action_val = action_range(1); 

    SOC_0 = .5;
    I_input = -25.7;
    
    [soc_state_values, ~, ~] = Discretization_Dict([min_state_val, max_state_val], num_states);
    [action_values, ~, ~] = Discretization_Dict([min_action_val, max_action_val], num_actions);

    [state_value, state_index] = Discretize_Value(SOC_0, soc_state_values);     

    spme_params = load("C:\Users\Indy-Windows\Desktop\SPMe_opt3_const_vars.mat");

    full_sim = 0;
    init_flag = 1; 

    xn = 0; 
    xp = 0; 
    xe = 0; 
    Sepsi_p = 0; 
    Sepsi_n = 0; 
    
    action_list = [];
    soc_list = [];
    soc_list = [soc_list, SOC_0];

    for t =1:1:1800
        
        [~, action_index] = max(Q_table(state_index, :));

        I_input = action_values(action_index);
        
        action_list = [action_list, I_input];

    if t ==1
        init_flag =1; 
        xn = 1.0e+11*[9.3705; 0; 0];
        xp = 1.0e+11*[4.5097; 0; 0];
        xe = [0;0];
        Sepsi_p = [0; 0; 0];
        Sepsi_n = [0; 0; 0];
    end 

            [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new,...
                dV_dEpsi_sp, soc_new, V_term, ...
                theta, docv_dCse, done_flag] = SPMe_step( xn, xp, xe, Sepsi_p, Sepsi_n,  I_input, full_sim, init_flag, spme_params);

        soc = soc_new(1);

        xn =xn_new;
        xp =xp_new;
        xe =xe_new;
        Sepsi_p =Sepsi_p_new;
        Sepsi_n =Sepsi_n_new;

        % Discretize the Resulting SOC
        [new_state_value, new_state_index] = Discretize_Value(soc, soc_state_values);

        state_value = new_state_value;
        state_index = new_state_index;
        soc_list = [soc_list,state_value ];

    end     
end 
