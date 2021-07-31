% Reset environment to initial state and output initial observation
function [InitialObservation, LogSig] = reset()

    xn = 1.0e+11.*[9.3705; 0; 0];
    xp = 1.0e+11.*[4.5097; 0; 0];
    xe = [0;0];
    Sepsi_p = [0; 0; 0];
    Sepsi_n = [0; 0; 0];
    soc = .5;
            
    InitialObservation = soc; 
    %     LogSig = [xn; xp; xe; Sepsi_p; Sepsi_n; soc];

    LogSig.xn = xn;
    LogSig.xp = xp;
    LogSig.xe = xe;
    LogSig.Sepsi_p = Sepsi_p;
    LogSig.Sepsi_n = Sepsi_n;
    LogSig.soc = soc;

%     disp(LogSig)


end