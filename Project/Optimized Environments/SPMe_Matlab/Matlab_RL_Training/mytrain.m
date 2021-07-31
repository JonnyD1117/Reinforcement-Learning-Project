function [newAgent,trainStats] = mytrain(agent,env)
% [NEWAGENT,TRAINSTATS] = mytrain(AGENT,ENV) train AGENT within ENVIRONMENT
% with the training options specified on the Train tab of the Reinforcement Learning Designer app.
% mytrain returns trained agent NEWAGENT and training statistics TRAINSTATS.

% Reinforcement Learning Toolbox
% Generated on: 30-Jul-2021 16:41:40

%% Create training options
trainOptions = rlTrainingOptions();
trainOptions.StopTrainingCriteria = "EpisodeCount";
trainOptions.Verbose = 1;

%% Make copy of agent
newAgent = copy(agent);

%% Perform training
trainStats = train(newAgent,env,trainOptions);
