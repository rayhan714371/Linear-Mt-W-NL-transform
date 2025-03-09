%% System Setup
% System matrices (from Section IV of the paper)
A = [1       0       0.995*0.01  0;  
     0       1       0           0.9975*0.01;  
     0       0       0.9975      0;  
     0       0       0           0.9975];  

B = [0.5*(0.01)^2   0; 
     0              0.5*(0.01)^2; 
     0.01           0; 
     0              0.01];  

C = eye(4);  % Measurement matrix

% Redesign observer gain L for stability using pole placement
desired_poles = [0.95, 0.9, 0.85, 0.8];  % Ensure eigenvalues < 1
L = place(A', C', desired_poles)';       % Stable observer gain

% Initial conditions
x0 = [10; -20; 30; -10];  % Initial state [Px, Py, Vx, Vy]
x = x0;                    % True state
x_hat = zeros(4,1);        % Estimated state

% Simulation parameters
N = 10000;                    % Simulation steps (100 seconds)
dt = 0.01;                    % Sampling time
time = (0:N-1)*dt;            % Time vector

% Preallocate variables
e_history = zeros(4,N);    % Estimation error history
y_history = zeros(4,N);    % Measurement history
flag = zeros(1,N);         % Attack detection flags

%% Proactive Defense Parameters
% Nonlinear transformation (scaled to avoid domain errors)
f = @(y) tanh(y/10);       % Gentle nonlinearity to keep inputs in [-1, 1]
f_inv = @(y) 10*atanh(y);  % Safe inverse transformation

% Noise bounds (UBB assumption)
epsilon = 0.02;  % Measurement noise bound ||v_k|| ≤ ε
delta = 0.05;    % Process noise bound ||ω_k|| ≤ δ

% Detection threshold (Theorem 2)
r_threshold = norm(C)*(delta + epsilon) + epsilon;  % ≈ 0.15

%% Main Simulation Loop
for k = 1:N
    % --- System Dynamics ---
    omega = delta*randn(4,1);      % Process noise (||ω|| ≤ δ)
    v = epsilon*randn(4,1);        % Measurement noise (||v|| ≤ ε)
    y = C*x + v;                   % True measurement
    
    % --- FDI Attack Injection (20-30 seconds) ---
    if k*dt >= 20 && k*dt <= 30
        attack_signal = [8; -8; 4; -4];  % Stealthy attack aligned with MTD
        y = y + attack_signal; 
    end
    
    % --- Proactive Defense Transformations ---
    % 1. Nonlinear transformation
    y_prime = f(y);          
    
    % 2. MTD scaling (moderate time-varying)
    Gamma_k = diag([1.5 + 0.5*sin(k), 1.5 + 0.5*cos(k), 2, 2]);  
    y_M = Gamma_k * y_prime; 
    
    % 3. Remove MTD & nonlinearity (receiver side)
    y_recovered = f_inv(Gamma_k \ y_M);  
    
    % --- State Estimation ---
    x_hat = A*x_hat + L*(y_recovered - C*x_hat);  
    
    % --- Residual Calculation ---
    r = norm(y_recovered - C*x_hat);  
    
    % --- Attack Detection ---
    if r > r_threshold
        flag(k) = 1;  % Set attack flag to 1 if residual exceeds threshold
    else
        flag(k) = 0;  % Set attack flag to 0 otherwise
    end
    
    % --- Update System State ---
    x = A*x + omega;          
    e_history(:,k) = x - x_hat;  % Store estimation error
    y_history(:,k) = y;          % Store measurement
end

%% Visualization (Publication-Ready Plots)
figure('Position', [100 100 1200 800], 'Color', 'white');

% 1. Position States (Px and Py)
subplot(3,1,1);
plot(time, y_history(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(time, y_history(2,:), 'r--', 'LineWidth', 1.5);
title('Position States (P_x and P_y)', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Position (m)', 'FontSize', 12);
legend('P_x', 'P_y', 'Location', 'northeast');
grid on;
xlim([0 100]);

% 2. Estimation Error (Px)
subplot(3,1,2);
plot(time, e_history(1,:), 'b-', 'LineWidth', 1.5);
title('Estimation Error (P_x)', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Error (m)', 'FontSize', 12);
grid on;
xlim([0 100]);

% 3. Attack Detection Flag with Shaded Regions
subplot(3,1,3);
stairs(time, flag, 'r', 'LineWidth', 1.5);
title('Attack Detection Flag', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Flag', 'FontSize', 12);
ylim([-0.1 1.5]);
yticks([0 1]);
yticklabels({'No Attack', 'Attack'});
grid on;
xlim([0 100]);

% Highlight attack periods
hold on;
attack_periods = (flag == 1);
area(time, attack_periods, 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
hold off;