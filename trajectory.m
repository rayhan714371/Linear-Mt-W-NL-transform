% System parameters
dt = 0.01;
gamma = 0.25;

% State-space matrices
A = [1, 0, (1 - 0.5*dt)*dt, 0;
     0, 1, 0, (1 - 0.5*gamma*dt)*dt;
     0, 0, 1 - gamma*dt, 0;
     0, 0, 0, 1 - gamma*dt];
 
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];

% Controller gain
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];

% Initial state
x0 = [10; -20; 30; -10];

% Simulation time (adjust as needed)
simTime = 5; % seconds
numSteps = simTime / dt;

% Initialize states
x = x0;
Px = zeros(1, numSteps + 1);
Py = zeros(1, numSteps + 1);
Px(1) = x0(1);
Py(1) = x0(2);

% Simulate closed-loop system
for k = 1:numSteps
    u = -K * x;                % Control input
    x = A * x + B * u;         % Update state
    Px(k+1) = x(1);            % Store P_x
    Py(k+1) = x(2);            % Store P_y
end

% Plotting
figure;
plot(Px, Py, 'b-', 'LineWidth', 1.5);
xlabel('P_x (m)');
ylabel('P_y (m)');
title('UAS Trajectory');
grid on;

% Adjust axes 
xlim([-2 14]);                   % X-axis limits
ylim([-25 5]);                 % Y-axis limits