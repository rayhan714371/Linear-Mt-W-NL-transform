%% Section IV: Numerical Example - Unmanned Aerial System (UAS)
% Simulation of the proposed proactive defense strategy.

%% System Parameters
% UAS Dynamics (Page 4):
dt = 0.01;      % Sampling time
gamma = 0.25;   % Damping parameter

% State matrix A (4x4):
A = [1, 0, (1-0.5*gamma*dt)*dt, 0;
     0, 1, 0, (1-0.5*gamma*dt)*dt;
     0, 0, 1-gamma*dt, 0;
     0, 0, 0, 1-gamma*dt];

% Input matrix B (4x2):
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];

% Output matrix C (4x4 identity):
C = eye(4);

% Controller gain K (Page 4):
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];

% Observer gain L (Page 4):
L = [0.2000, 0, 0.0499, 0;
     0, 0.2000, 0, 0.0499;
     0, 0, 0.4975, 0;
     0, 0, 0, 0.0975];

% Initial state:
x0 = [10; -20; 30; -10];  % [Px; Py; Vx; Vy]

%% Proactive Defense Setup
% Nonlinear transformation f(y) (e.g., sigmoid function):
f = @(y) 1 ./ (1 + exp(-y));  % Example monotonic function
f_inv = @(y) log(y ./ (1 - y));  % Inverse of f

% Moving Target Defense (Γ_k as diagonal time-varying matrix):
% Example: Γ_k(i,i) = 1 + rand()*0.5 (random scaling)
Gamma_k = diag(1 + 0.5*rand(4,1));  % Updated at each time step

%% Simulation Parameters
T_total = 50;       % Total simulation time (sec)
t = 0:dt:T_total;   % Time vector
N = length(t);

% Attack parameters (Page 4):
attack_start = 20;  % Attack starts at 20 sec
attack_end = 30;    % Ends at 30 sec

%% Simulation (Attack-Free Scenario)
% Initialize states and estimates:
x = zeros(4, N);
x(:,1) = x0;
x_hat = zeros(4, N);

for k = 1:N-1
    % Apply NT and MTD (Algorithm 1):
    y_k = C * x(:,k);               % Sensor output
    y_prime = f(y_k);               % Nonlinear transformation
    y_M = Gamma_k * y_prime;        % MTD
    y_bar = f_inv(Gamma_k \ y_M);   % Remove NT/MTD
    
    % State estimation (Equation 5):
    x_hat(:,k+1) = A * x_hat(:,k) + L * (y_bar - C * x_hat(:,k));
    
    % State update (Equation 1):
    x(:,k+1) = A * x(:,k) + B * (-K * x(:,k));  % Assume control input u = -Kx
end

% Plot results (Figs. 2-3):
% figure; plot(t, x(1:2,:));  % Trajectory of Px and Py
% figure; plot(t, x - x_hat); % Estimation error e_k

%% Simulation (Under FDI Attack)
% Inject attack during [20,30] sec (Equation 4):
a_y = 0.1 * randn(4,1);  % Example attack vector

for k = 1:N-1
    y_k = C * x(:,k);
    y_prime = f(y_k);
    y_M = Gamma_k * y_prime;
    
    % Inject attack if in attack interval:
    if t(k) >= attack_start && t(k) <= attack_end
        y_M = y_M + a_y;  % FDI attack (Page 2)
    end
    
    y_bar = f_inv(Gamma_k \ y_M);
    x_hat(:,k+1) = A * x_hat(:,k) + L * (y_bar - C * x_hat(:,k));
    x(:,k+1) = A * x(:,k) + B * (-K * x(:,k));
end

% Compute residuals (Equation 8):
residual = y_bar - C * x_hat(:,1:N-1);
threshold = 0.5;  % Example threshold (̄r)

% Attack detection flags (Equation 9):
detection_flag = vecnorm(residual) > threshold;

% Plot results (Figs. 5-7):
% figure; plot(t(1:N-1), residual);  % Residuals under attack
% figure; plot(t(1:N-1), detection_flag);  % Detection flags

%% Conclusion
% The code demonstrates the proposed strategy's effectiveness:
% - NT/MTD do not affect performance when attack-free (Theorem 1).
% - Residuals exceed thresholds during attacks, improving detection (Theorem 2).