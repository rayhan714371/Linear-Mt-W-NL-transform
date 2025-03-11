% System parameters (pendulum)
g = 9.81;   % Gravity (m/s²)
L = 1.0;    % Pendulum length (m)
b = 0.1;    % Damping coefficient
m = 1.0;    % Mass (kg)
dt = 0.01;  % Time step
simTime = 40; % Simulation time
numSteps = simTime / dt;

% Nonlinear state equations (discretized)
f_nonlinear = @(x, u) [
    x(1) + dt * x(2);  % θ_{k+1} = θ_k + dt * ω_k
    x(2) + dt * (-(g/L)*sin(x(1)) - dt*(b/m)*x(2) + dt*u/(m*L^2))  % ω_{k+1}
];

% Measurement matrix (C)
C = eye(2); % Measure θ and ω

% Controller gain (designed empirically; original K is not optimal)
K = [10, 2]; % Gains for θ and ω

% Observer gain (Luenberger observer)
L_obs = [0.5; 0.5]; % Observer gains

% Initial states
x_true = [pi/4; 0];      % True state: [θ; ω]
x_hat = [0; 0];          % Estimated state

% Nonlinear transformation and moving target
f = @(y) y.^3;           % Nonlinear transformation
f_inv = @(y) sign(y).*abs(y).^(1/3);
Gamma_k = @(k) diag(1 + 0.5*sin(k/10)*ones(2,1)); % Time-varying Γ

% Attack parameters
attack_start = 20/dt;
attack_end = 30/dt;
a_y = [0.5; 0];          % Bias attack on θ measurement

% Storage variables
theta = zeros(1, numSteps + 1);
omega = zeros(1, numSteps + 1);
flags = zeros(1, numSteps + 1);
residuals = zeros(2, numSteps + 1);
time = 0:dt:simTime;

% Simulation loop
for k = 1:numSteps
    % Control input (nonlinear system requires careful tuning)
    u = -K * (x_true - [0; 0]); % Stabilize to upright position
    
    % Update true state (nonlinear dynamics)
    x_true = f_nonlinear(x_true, u) + 0.1*randn(2,1); % Add noise
    
    % Measurement with noise
    y_true = C * x_true + 0.05*randn(2,1);
    
    %--- Proactive defense ---
    y_prime = f(y_true);          % Nonlinear transformation
    y_M = Gamma_k(k) * y_prime;   % Moving target
    
    % FDI attack (20-30s)
    if k >= attack_start && k <= attack_end
        y_M_attacked = y_M + a_y;
    else
        y_M_attacked = y_M;
    end
    
    % Inverse transformations
    y_prime_recovered = Gamma_k(k) \ y_M_attacked;
    y_bar = f_inv(y_prime_recovered);
    
    % State estimation (Luenberger observer)
    x_hat = f_nonlinear(x_hat, u) + L_obs .* (y_bar - C * x_hat);
    
    % Residual and detection
    residuals(:, k+1) = y_bar - C * x_hat;
    if norm(residuals(:, k+1)) > 0.3
        flags(k+1) = 1;
    end
    
    % Store states
    theta(k+1) = x_true(1);
    omega(k+1) = x_true(2);
end

% Plot results
figure;
subplot(3,1,1);
plot(time, theta, 'b', 'LineWidth', 1.5);
ylabel('θ (rad)');
title('Nonlinear Pendulum: Angle (θ)');

subplot(3,1,2);
plot(time, omega, 'b', 'LineWidth', 1.5);
ylabel('ω (rad/s)');
title('Angular Velocity (ω)');

subplot(3,1,3);
stairs(time, flags, 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Attack Flag');
ylim([-0.1, 1.1]);