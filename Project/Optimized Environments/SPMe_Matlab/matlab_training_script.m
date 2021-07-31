clear all 
clc
%% Define SPMe Environment via rlFunctionEnv

% Initialize Observation settings
    ObservationInfo = rlNumericSpec([1 1]);
    ObservationInfo.Name = 'SPMeBatteryEnv States';
    ObservationInfo.Description = 'soc';

% Initialize Action settings   
    action_space = linspace(-25.7, 25.7, 45);
    ActionInfo = rlFiniteSetSpec(action_space);
    ActionInfo.Name = 'SPMeBatteryEnv Action';

%%% Create Environment 
%     env = SPMeBatteryEnv();
    env = rlFunctionEnv(ObservationInfo, ActionInfo, "stepfcn", "resetfcn");

% %%% Create DQN Agent
%     obs_info = getObservationInfo(env);
%     act_info = getActionInfo(env); 
% 
%     agentOpts = rlDQNAgentOptions(...
%         'UseDoubleDQN',false, ...    
%         'TargetUpdateMethod',"periodic", ...
%         'TargetUpdateFrequency',4, ...   
%         'ExperienceBufferLength',100000, ...
%         'DiscountFactor',0.99, ...
%         'MiniBatchSize',256);
% 
%     agent = rlDQNAgent(obs_info, act_info ,agentOpts);
% 
% %%% Create training options
%     trainOptions = rlTrainingOptions();
%     trainOptions.StopTrainingCriteria = "EpisodeCount";
% 
% %%% Perform training
%     trainStats = train(agent,env,trainOptions);