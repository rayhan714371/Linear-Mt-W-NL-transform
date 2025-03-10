% Simulation parameters
dt = 0.01;
gamma = 0.25;
sim_time = 35; % seconds
N = round(sim_time / dt);
t = 0:dt:(sim_time - dt);

% System matrices
A = [1, 0, (1-0.5*dt)*dt, 0;
     0, 1, 0, (1-0.5*gamma*dt)*dt;
     0, 0, 1 - gamma*dt, 0;
     0, 0, 0, 1 - gamma*dt];
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];
C = eye(4);
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];
L = [0.2000, 0, 0.0499, 0;
     0, 0.2000, 0, 0.0499;
     0, 0, 0.4975, 0;
     0, 0, 0, 0.0975];

% Initial conditions
x = [10; -20; 30; -10];
x_hat = x; % Initial state estimate

% Noise bounds (user-defined)
delta = 0.1;    % Process noise bound
epsilon = 0.1;  % Measurement noise bound

% Attack parameters
attack_start = 20;
attack_end = 30;
attack_indices = find(t >= attack_start & t <= attack_end);
a_k = [5; 5; 0; 0]; % Attack vector

% Defense mechanisms
f = @(y) 1.5 * y;        % Nonlinear transformation
f_inv = @(y) y / 1.5;    % Inverse transformation
rng(0); % Reproducible random Gamma matrices
Gamma = zeros(4,4,N);
for k = 1:N
    Gamma(:,:,k) = diag(0.9 + 0.2*rand(4,1)); % Random diagonal
end

% Detection threshold (adjust based on normal operation)
r_bar = 1.5;

% Data storage
Px = zeros(1,N); Py = zeros(1,N);
error = zeros(4,N); flags = zeros(1,N);
residuals = zeros(1,N);

% Simulation loop
for k = 1:N
    % Control input
    u = -K * x_hat;
    
    % Process noise (bounded)
    omega = delta * (2*rand(4,1) - 1);
    x = A * x + B * u + omega;
    
    % Measurement noise (bounded)
    v = epsilon * (2*rand(4,1) - 1);
    y_k = C * x + v;
    
    % Inject FDI attack
    if ismember(k, attack_indices)
        y_k = y_k + a_k;
    end
    
    % Apply defense: nonlinear + MT
    y_prime = f(y_k);
    y_M = Gamma(:,:,k) * y_prime;
    y_bar = f_inv(Gamma(:,:,k) \ y_M);
    
    % State estimation
    x_hat = A * x_hat + L * (y_bar - C * x_hat);
    
    % Residual and detection
    r_a = y_bar - C * x_hat;
    residuals(k) = norm(r_a);
    flags(k) = residuals(k) > r_bar;
    
    % Save data
    Px(k) = x(1); Py(k) = x(2);
    error(:,k) = x - x_hat;
end

% Plotting
figure;

% Trajectories under attack
subplot(2,2,1);
plot(t, Px, 'b', t, Py, 'r');
hold on; xline(20, 'k--'); xline(30, 'k--');
title('Trajectories of P_x and P_y under FDI Attack');
xlabel('Time (s)'); ylabel('Position');
legend('P_x', 'P_y', 'Attack Start', 'Attack End');

% Estimation error
subplot(2,2,2);
plot(t, error(1,:), 'b', t, error(2,:), 'r');
title('State Estimation Error (P_x, P_y)');
xlabel('Time (s)'); ylabel('Error');
legend('e_{P_x}', 'e_{P_y}');

% Detection flags
subplot(2,2,3);
stairs(t, flags); ylim([-0.1, 1.1]);
title('Attack Detection Flags');
xlabel('Time (s)'); ylabel('Flag');

% Residual vs threshold
subplot(2,2,4);
plot(t, residuals, 'k', t, r_bar*ones(size(t)), 'r--');
title('Residual Norm vs Threshold');
xlabel('Time (s)'); ylabel('Norm');
legend('Residual', 'Threshold');

% Adjust layout
set(gcf, 'Position', [100, 100, 1200, 800]);