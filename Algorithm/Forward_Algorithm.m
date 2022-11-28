% INITIALIZE HMM MODEL SPECIFICATION
syms A B;
S = [A B]; % hidden states

syms x y z; 
V = [x y z]; % observable states

pi = [0.8 0.2]; % initial probability
A = [0.7 0.3; 0.4 0.6]; % state transition probability
B = [0.2 0.4 0.4;0.5 0.4 0.1]; % emission probability

O = [x z x]; % observable sequence

N = length(S); % number of distinct states
T = length(O); % number of data in observable sequence

alfa = zeros(N,T); % forward variable
P_O_lambda = 0; % probabability of observation sequence based on model (lambda)

% START ALGORITHM

% 1. Initialization
for i = 1:N
    alfa(i,1) = pi(i)*B(i,V(:)==O(1));
end

% 2. Induction
for t = 1:(T-1)
    for j = 1:N
        sigma_alfa_a = 0;
        for i = 1:N
            sigma_alfa_a = sigma_alfa_a + alfa(i,t)*A(i,j);
        end
        alfa(j,t+1) = sigma_alfa_a*B(j,V(:)==O(t+1));
    end
end

% 3. Termination
for i = 1:N
    P_O_lambda = P_O_lambda + alfa(i,T);
end


