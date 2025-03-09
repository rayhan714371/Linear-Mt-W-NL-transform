%% ============================================================  
%% PARAMETERS (From Section IV of the Paper)  
%% ============================================================  
% System Dynamics:  
%   - Sampling time: dt = 0.01  
%   - Damping parameter: gamma = 0.25  
%   - State matrix A (4x4):  
%     [1, 0, (1-0.5*gamma*dt)*dt, 0;  
%      0, 1, 0, (1-0.5*gamma*dt)*dt;  
%      0, 0, 1-gamma*dt, 0;  
%      0, 0, 0, 1-gamma*dt]  
%   - Input matrix B (4x2):  
%     [0.5*dt^2, 0;  
%      0, 0.5*dt^2;  
%      dt, 0;  
%      0, dt]  
%   - Output matrix C: eye(4) (4x4 identity matrix)  
%   - Controller gain K:  
%     [40.0400, 0, 29.5498, 0;  
%      0, 20.2002, 0, 68.7490]  
%   - Observer gain L:  
%     [0.2000, 0, 0.0499, 0;  
%      0, 0.2000, 0, 0.0499;  
%      0, 0, 0.4975, 0;  
%      0, 0, 0, 0.0975]  
%   - Initial state: x0 = [10; -20; 30; -10] (Px, Py, Vx, Vy)  

% Attack Configuration:  
%   - Attack time interval: [20 sec, 30 sec]  
%   - Attack vector: Random FDI injection (e.g., a_y = 0.1 * randn(4,1))  

% Proactive Defense Setup:  
%   - Nonlinear transformation f(y):  
%     Example: Sigmoid function f(y) = 1 ./ (1 + exp(-y))  
%     Inverse: f_inv(y) = log(y ./ (1 - y))  
%   - Moving Target Γ_k:  
%     Diagonal matrix with entries Γ_k(i,i) = 1 + 0.5*rand()  
%   - Detection threshold: threshold = 0.5 (example value)  

%% ============================================================  
%% INSTRUCTIONS (Algorithm Steps from the Paper)  
%% ============================================================  
% 1. Initialize system matrices (A, B, C), controller (K), observer (L), and initial state (x0).  
% 2. Define the nonlinear transformation function (f) and its inverse (f_inv).  
% 3. Generate a time-varying diagonal matrix Γ_k at each time step.  
% 4. Simulate the attack-free scenario:  
%    - Compute sensor output y_k = C * x.  
%    - Apply NT: y_prime = f(y_k).  
%    - Apply MTD: y_M = Γ_k * y_prime.  
%    - Revert NT/MTD: y_bar = f_inv(Γ_k \ y_M).  
%    - Update state estimate: x_hat(:,k+1) = A * x_hat(:,k) + L * (y_bar - C * x_hat(:,k)).  
% 5. Simulate the FDI attack scenario:  
%    - Inject attack vector a_y into y_M during [20 sec, 30 sec].  
%    - Recover y_bar and compute residuals: r_k = y_bar - C * x_hat.  
% 6. Check detection flags: Trigger alarm if ||r_k|| > threshold.  
% 7. Plot results:  
%    - Trajectories (Px, Py)  
%    - State estimation errors (e_k)  
%    - Residuals and detection flags  

%% ============================================================  