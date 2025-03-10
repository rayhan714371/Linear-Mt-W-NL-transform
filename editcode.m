% Parameters
dt = 0.01; % Sampling time
gamma = 0.25; % Damping parameter
A = [1 0 (1-0.5*dt)*dt 0; 0 1 0 (1-0.5*gamma*dt)*dt; 0 0 1-gamma*dt 0; 0 0 0 1-gamma*dt];
B = [0.5*dt^2 0; 0 0.5*dt^2; dt 0; 0 dt];
C = eye(4); % Measurement matrix
x0 = [10 -20 30 -10]'; % Initial state

% Attack model parameters
attack_threshold = 1.0; % Residual threshold for detection
attack_start = 20; % Attack start time
attack_end = 30; % Attack end time

% Nonlinear transformation function (bijective function)
f = @(x) tanh(x); % Example nonlinear function
f_inv = @(x) atanh(x); % Inverse function

% Simulation setup
N = 100; % Number of time steps
xk = x0; % Initial state
x_hat = zeros(4, 1); % Initial state estimate
L = [0.2 0 0.0499 0; 0 0.2 0 0.0499; 0 0 0.4975 0; 0 0 0 0.0975]; % Observer gain
K = [40.0400 0 29.5498 0; 0 20.2002 0 68.7490]; % Controller gain
y = C * xk; % Measurement

% Arrays for storing results
xk_all = zeros(4, N);
y_all = zeros(4, N);
x_hat_all = zeros(4, N);
rak_all = zeros(1, N);
attack_flag = zeros(1, N);

% Main loop for simulation
for k = 1:N
    % System dynamics (with attack between [attack_start, attack_end])
    if k >= attack_start && k <= attack_end
        % Injecting False Data Injection Attack
        y = C * xk + 0.1 * randn(4, 1) + 2 * sin(0.1*k)'; % Malicious attack on the system
    else
        y = C * xk + 0.1 * randn(4, 1); % Normal measurement noise
    end
    
    % Apply nonlinear transformation and moving target defense
    y_prime = f(y); % Apply nonlinear transformation
    gamma_k = diag(1 + 0.1*randn(4, 1)); % Random moving target defense matrix
    y_Mk = gamma_k * y_prime; % Apply moving target
    
    % State estimation using observer
    x_hat = A * x_hat + L * (y_Mk - C * x_hat); % Observer update
    
    % Residual for attack detection
    rak = norm(y_Mk - C * x_hat); % Residual computation
    
    % Detection logic
    if rak > attack_threshold
        attack_flag(k) = 1; % Attack detected
    else
        attack_flag(k) = 0; % No attack
    end
    
    % Store results
    xk_all(:, k) = xk;
    y_all(:, k) = y;
    x_hat_all(:, k) = x_hat;
    rak_all(k) = rak;
    
    % System dynamics for next time step
    xk = A * xk + B * randn(2, 1); % Process noise added
end

% Plot the results similar to Figures 4, 5, and 6
figure;

% Figure 4: State trajectory Px and Py (positions)
subplot(3, 1, 1);
plot(1:N, xk_all(1, :), 'r', 'LineWidth', 2); hold on;
plot(1:N, xk_all(2, :), 'b', 'LineWidth', 2);
xlabel('Time step');
ylabel('State (Px, Py)');
title('Figure 4: Trajectory of Px and Py');
legend('Px', 'Py');
grid on;

% Figure 5: Estimation error ek
subplot(3, 1, 2);
plot(1:N, xk_all(1, :) - x_hat_all(1, :), 'r', 'LineWidth', 2); hold on;
plot(1:N, xk_all(2, :) - x_hat_all(2, :), 'b', 'LineWidth', 2);
xlabel('Time step');
ylabel('Estimation error (ek)');
title('Figure 5: Estimation Error ek');
legend('e_Px', 'e_Py');
grid on;

% Figure 6: Attack detection flag
subplot(3, 1, 3);
plot(1:N, attack_flag, 'k', 'LineWidth', 2);
xlabel('Time step');
ylabel('Attack Detected');
title('Figure 6: Attack Detection Results');
legend('Flag (1 = Attack, 0 = No Attack)');
grid on;