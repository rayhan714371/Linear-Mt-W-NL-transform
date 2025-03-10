% Define system parameters
dt = 0.01; % Time step
gamma = 0.25; % Gamma parameter
N = 100; % Number of time steps
r_bar = 1; % Detection threshold

% System matrices
A = [1, 0, (1-0.5*dt)*dt, 0; 
     0, 1, 0, (1-0.5*gamma*dt)*dt; 
     0, 0, 1-gamma*dt, 0; 
     0, 0, 0, 1-gamma*dt]; 

B = [0.5*dt^2, 0; 
     0, 0.5*dt^2; 
     dt, 0; 
     0, dt]; 

C = eye(4);

% Controller and observer gains
K = [40.0400, 0, 29.5498, 0; 
     0, 20.2002, 0, 68.7490]; 

L = [0.2000, 0, 0.0499, 0; 
     0, 0.2000, 0, 0.0499; 
     0, 0, 0.4975, 0; 
     0, 0, 0, 0.0975];

% Initial state and input
x_0 = [10; -20; 30; -10]; 
u_k = [0; 0]; 

% Initialize variables
x_hat = x_0; % State estimate
x = x_0; % Actual state
P_x = x(1); P_y = x(2); V_x = x(3); V_y = x(4); 
y_k = C*x + 0.1*randn(4,1); % Initial sensor measurement
flag = zeros(N,1); % Attack detection flag
r_a_k = zeros(N,1); % Residual
P_x_vals = zeros(N,1); % Store P_x for plotting
P_y_vals = zeros(N,1); % Store P_y for plotting
e_k_vals = zeros(N,4); % State estimation error

% Nonlinear transformation and moving target parameters
Gamma_k = diag([1,1,1,1]); % Identity for simplicity, could vary
f = @(y) y; % Identity transformation for simplicity (modify if needed)

% Run the simulation
% Run the simulation
for k = 1:N
    % Apply nonlinear transformation and moving target
    y_prime_k = f(y_k);
    y_M_k = Gamma_k * y_prime_k;
    y_bar_k = inv(Gamma_k) * y_M_k;
    
    % State estimate update
    x_hat = A*x_hat + L*(y_bar_k - C*x_hat);
    
    % Calculate residual
    r_a_k(k) = norm(y_bar_k - C*x_hat);
    
    % Attack detection (simple flagging)
    if r_a_k(k) > r_bar
        flag(k) = 1;
    end
    
    % Save state and error values
    P_x_vals(k) = x(1);
    P_y_vals(k) = x(2);
    e_k_vals(k,:) = (x_hat - x)'; % Corrected to transpose the state estimation error
    
    % State dynamics with FDI attack injection
    if k >= 20 && k <= 30
        y_k = C*x + 1.5*randn(4,1); % Injecting FDI attack by corrupting sensor output
    else
        y_k = C*x + 0.1*randn(4,1); % Normal sensor noise
    end
    
    % Update state using system dynamics
    x = A*x + B*u_k + [0.1*randn; 0.1*randn; 0.1*randn; 0.1*randn]; % System dynamics
end

% Plot the results
figure;
subplot(2,2,1);
plot(P_x_vals, P_y_vals, '-o');
title('Fig. 1: Trajectory of P_x and P_y in the attack-free');
xlabel('P_x (m)');
ylabel('P_y (m)');

subplot(2,2,2);
plot(1:N, e_k_vals(:,1), 'r-', 1:N, e_k_vals(:,2), 'g-', 1:N, e_k_vals(:,3), 'b-', 1:N, e_k_vals(:,4), 'k-');
title('Fig. 2: Estimation error e_k in the attack-free');
xlabel('Time k');
ylabel('The State Estimation Error e_k');
legend('e_{P_x}', 'e_{P_y}', 'e_{V_x}', 'e_{V_y}');

subplot(2,2,3);
plot(1:N, e_k_vals(:,1), 'r-', 1:N, e_k_vals(:,2), 'g-', 1:N, e_k_vals(:,3), 'b-', 1:N, e_k_vals(:,4), 'k-');
title('Fig. 3: The estimation error e_k under FDI attacks');
xlabel('Time k');
ylabel('The State Estimation Error e_k');
legend('e_{P_x}', 'e_{P_y}', 'e_{V_x}', 'e_{V_y}');

subplot(2,2,4);
plot(1:N, flag, 'r-', 'LineWidth', 1.5);
title('Fig. 4: The attack detection results');
xlabel('Time k');
ylabel('The detection results');
ylim([0 1]);

% Show all plots
sgtitle('Simulation Results for Active Attack Defense Strategy');