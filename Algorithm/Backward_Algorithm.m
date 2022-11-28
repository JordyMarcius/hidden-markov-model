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

beta = zeros(N,T); % backward variable

% START ALGORITHM

% 1. Initialization
for i = 1:N
    beta(i,T) = 1;
end

% 2. Induction
for t = T-1:-1:1
    for i = 1:N
        sigma_a_b_beta = 0;
        for j = 1:N
            sigma_a_b_beta = sigma_a_b_beta + A(i,j)*B(j,V(:)==O(t+1))*beta(j,t+1);
        end
        beta(i,t) = sigma_a_b_beta;
    end
end



