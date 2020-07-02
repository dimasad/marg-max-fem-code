%% Setup analysis

data_dir = strcat('data', filesep, 'mc');
config = load([data_dir, filesep, 'config']);
data_files = dir(strcat(data_dir, filesep, 'exp*.mat'));
nx = config.nx;

%% Loop through all experiments
for data_file = data_files'
    disp(data_file.name);
    
    %% Load data
    data = load([data_file.folder, filesep, data_file.name]);
    d = iddata(data.y, data.u);
    dv = d(1:end/2);
    de = d(end/2+1:end);
    
    %% Estimate
    s1opt = n4sidOptions('EnforceStability', true, 'N4Weight', 'MOESP');
    s1 = n4sid(de, nx, 'Feedthrough', true, s1opt);
    
    s2opt = ssestOptions('InitialState', 'estimate', 'SearchMethod', 'gn');
    s2opt.SearchOptions.MaxIterations = 100;
    s2opt.SearchOptions.Tolerance = 0;
    s2 = ssest(de, s1, s2opt);    
    
    %% Validate with complementary data (dv)
    copt = compareOptions('InitialCondition', 'z');
    [~, fit] = compare(dv, s1, s2, copt);
    
    %% Get balanced realization
    [s1bal, gram, T] = balreal(s1);
    s1bal = idss(s1bal);
    s1bal.K = T * s1.K;
    s1bal.x0 = T * s1.x0;
    
    %% Save
    sys = [sys_struct(s1), sys_struct(s2)];
    sys(1).fit = fit{1};
    sys(2).fit = fit{2};
    
    guess = sys_struct(s1bal);
    guess.gram = gram;
    guess.x0 = s1bal.x0;   
    
    est_file = [data_dir, filesep, 'estim_', data_file.name];
    save(est_file, 'guess', 'sys');
end

function s = sys_struct(sys)
s = struct;
s.A = sys.A;
s.B = sys.B;
s.C = sys.C;
s.D = sys.D;
s.L = sys.K;
end