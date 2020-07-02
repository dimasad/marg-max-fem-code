%%%% Data generation for Monte-Carlo empirical study %%%%
%% Setup random number generator and define parameters
rng_settings = rng(0);

nx = 5;
nu = 2;
ny = 2;
nw = nx;

N = 1000;

std_u = 1;
std_w = 0.05;
std_v = 0.2;

nexp = 250;

%% Create data folder
start = datetime('now');
start_str = string(start, 'yyyy-MM-dd_HH''h''mm''m''ss''s''');
data_dir = strcat('data', filesep, 'mc_', start_str);
mkdir(data_dir);

%% Save config
config_file = strcat(data_dir, filesep, 'config');
config = struct;
config.N = N;
config.nx = nx;
config.nu = nu;
config.ny = ny;
config.nw = nw;
config.std_u = std_u;
config.std_w = std_w;
config.std_v = std_v;
config.rng_settings = rng_settings;
config.nexp = nexp;
save(config_file, '-struct', 'config')

%% Run experiments

for i=1:nexp
    disp(['experiment ', num2str(i)])
    %% Create system
    rng_settings = rng;

    while true
        sys = drss(nx, ny, nu + nw);
        sys.d(:, nu+1:end) = 0;
        
        if all(abs(pole(sys)) < 0.9999)
            break
        end
    end
    
    u = std_u * randn(N, nu);
    w = std_w * randn(N, nw);
    v = std_v * randn(N, ny);
    
    [yclean, ~, x] = lsim(sys, [u, w]);
    y = yclean + v;
    
    %% Save
    data = struct;
    data.rng_settings = rng_settings;
    
    data.u = u;
    data.w = w;
    data.v = v;
    data.y = y;
    data.x = x;
    
    data.A = sys.A;
    data.B = sys.B;
    data.C = sys.C;
    data.D = sys.D;
    
    data_file = strcat(data_dir, filesep, 'exp', num2str(i, '%04d'));
    save(data_file, '-struct', 'data'); 
end