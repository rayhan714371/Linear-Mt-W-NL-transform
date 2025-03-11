% System parameters
dt = 0.01;
gamma = 0.25;
simTime = 40; % Simulation time (includes attack interval [20,30])
numSteps = simTime / dt;

% Nonlinear state update function
nonlinear_state_update = @(x, u) [x(1) + dt * x(3) + 0.5 * dt^2 * u(1);
                                  x(2) + dt * x(4) + 0.5 * dt^2 * u(2);
                                  x(3) + dt * (-gamma * x(3) + u(1));
                                  x(4) + dt * (-gamma * x(4) + u(2))];

% Control input matrix (unchanged)
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];
C = eye(4);

% Controller and observer gains (unchanged)
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];
L = [0.2000, 0, 0.0499, 0;
     0, 0.2000, 0, 0.0499;
     0, 0, 0.4975, 0;
     0, 0, 0, 0.0975];

% Initial states
x = [10; -20; 30; -10]; % True state
hat_x = zeros(4,1);      % Estimated state (initialized to 0)

% Noise bounds (UBB)
delta = 0.1;  % Process noise bound
epsilon = 0.1; % Measurement noise bound

% Nonlinear transformation functions (element-wise cubic)
f = @(y) y.^3;               % y' = y^3
f_inv = @(y) sign(y).*abs(y).^(1/3); % Inverse: y = y'^(1/3)

% Moving target defense (Γ_k: time-varying diagonal matrix)
Gamma_k = @(k) diag(1 + 0.5*sin(k/10)*ones(4,1)); % Γ_k varies sinusoidally

% Attack parameters
attack_start = 20/dt;   % Start at 20 seconds (step index)
attack_end = 30/dt;     % End at 30 seconds
a_y = [5; 5; 0; 0];     % FDI attack vector (added to y^M_k)

% Threshold for residual detection
threshold = 0.8; % Scalar threshold for simplicity

% Storage variables
Px = zeros(1, numSteps + 1);
Py = zeros(1, numSteps + 1);
residual = zeros(4, numSteps + 1);
flags = zeros(1, numSteps + 1);
time = 0:dt:simTime;

% Initialize
Px(1) = x(1);
Py(1) = x(2);

%for estimation error
e_Px = zeros(1, numSteps + 1);
e_Py = zeros(1, numSteps + 1);
e_Vx = zeros(1, numSteps + 1);
e_Vy = zeros(1, numSteps + 1);

% Simulation loop
for k = 1:numSteps
    % Control input
    u = -K * x;
    
    % Update true state with process noise (bounded)
    omega = delta * (2*rand(4,1) - 1); % Random in [-delta, delta]
    x = nonlinear_state_update(x, u) + omega;
    
    % Measurement with noise
    v = epsilon * (2*rand(4,1) - 1);   % Random in [-epsilon, epsilon]
    y_k = C * x + v;
    
    %--- Proactive defense transformations ---
    % Apply nonlinear transformation: y' = f(y_k)
    y_prime = f(y_k);
    
    % Apply moving target: y^M_k = Γ_k * y'
    Gamma = Gamma_k(k); % Time-varying Γ
    y_M = Gamma * y_prime;
    
    %--- FDI attack injection ---
    if k >= attack_start && k <= attack_end
        y_M_attacked = y_M + a_y; % Inject attack
    else
        y_M_attacked = y_M;       % No attack
    end
    
    %--- Inverse transformations at controller ---
    y_prime_attacked = Gamma \ y_M_attacked; % Γ^{-1} * y_M_attacked
    y_bar = f_inv(y_prime_attacked);         % Recover measurement
    
    %--- State estimation ---
    hat_x = nonlinear_state_update(hat_x, L * (y_bar - C * hat_x));
    
    %--- Calculate estimation error ---
    e = x - hat_x;
    e_Px(k+1) = e(1);
    e_Py(k+1) = e(2);
    e_Vx(k+1) = e(3);
    e_Vy(k+1) = e(4);
    
    %--- Residual calculation ---
    residual(:, k+1) = y_bar - C * hat_x;
    
    %--- Attack detection ---
    if norm(residual(:, k+1)) > threshold
        flags(k+1) = 1; % Attack detected
    else
        flags(k+1) = 0; % No attack
    end
    
    % Store trajectory
    Px(k+1) = x(1);
    Py(k+1) = x(2);
end

figure;

% P_x estimation error
subplot(4,1,1);
plot(time, e_Px, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('P_x Error (m)');
title('State Estimation Errors');
grid on;

% P_y estimation error
subplot(4,1,2);
plot(time, e_Py, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('P_y Error (m)');
grid on;

% V_x estimation error
subplot(4,1,3);
plot(time, e_Vx, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('V_x Error (m/s)');
grid on;

% V_y estimation error
subplot(4,1,4);
plot(time, e_Vy, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
xlabel('Time (s)');
ylabel('V_y Error (m/s)');
grid on;

% Plot results
figure;

% Plot trajectory
subplot(2,1,1);
plot(Px, Py, 'b-', 'LineWidth', 1.5);
hold on;
scatter(Px(attack_start:attack_end), Py(attack_start:attack_end), 10, 'r', 'filled');
xlabel('P_x (m)');
ylabel('P_y (m)');
title('UAS Trajectory (Red: Attack Period)');
grid on;
set(gca, 'YDir', 'reverse');
xlim([0 14]);
ylim([-25 -5]);

% Plot detection flags
subplot(2,1,2);
stairs(time, flags, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Attack Flag');
title('Attack Detection Results');
ylim([-0.1 1.1]);
xlim([0 simTime]);
grid on;
