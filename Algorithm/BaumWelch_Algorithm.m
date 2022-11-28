% INITIALIZATION (MODEL SPECIFICATION)
syms A B;
S = [A B]; % hidden states

syms x y z; 
V = [x y z]; % observable states

pi = [0.8 0.2]; % initial probability
A = [0.7 0.3; 0.4 0.6]; % state transition probability
B = [0.2 0.4 0.4;0.5 0.4 0.1]; % emission probability

pi_hat = zeros(1,length(pi));
A_hat = zeros(length(A(:,1)),length(A));
B_hat = zeros(length(B(:,1)),length(B));

O = [x z x]; % observable sequence

N = length(S); % number of distinct hidden states
K = length(V); % number of distinct observable states
T = length(O); % number of data in observable sequence

alfa = zeros(N,T); % forward variable
beta = zeros(N,T); % backward variable
gamma = zeros(N,T); % probability of being state i in time t
xi = zeros(N^2,T-1); % probability of being state i in time t and state j in time t+1
P_O_lambda = 0; % probabability of observation sequence based on model (lambda)

% FORWARD ALGORITHM TO CALCULATE ALFA VARIABLE
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

% BACKWARD ALGORITHM TO CALCULATE BACKWARD VARIABLE
% 1. Initialization
for i = 1:N
    beta(i,T) = 1;
end

% 2. Induction
for t = T-1:-1:1
    for i = 1:N
        sigma_a_b_beta = 0;
        for j = 1:N
            sigma_a_b_beta = sigma_a_b_beta + A(i,j)*B(j,V(:)== O(t+1))*beta(j,t+1);
        end
        beta(i,t) = sigma_a_b_beta;
    end
end

% CALCULATE GAMMA VARIABLE
for t = 1:T
    for i = 1:N
        gamma(i,t) = (alfa(i,t)*beta(i,t))/P_O_lambda;
    end
end

% BAUM-WELCH ALGORITHM TO ESTIMATE MODEL SPECIFICATION
for t = 1:T-1
    for j = 1:N
        for i = 1:N
            xi(i,j,t) = (alfa(i,t)*A(i,j)*B(j,V(:)== O(t+1))*beta(j,t+1))/P_O_lambda;
        end
    end
end

% Initial probability estimated
for i = 1:N
    pi_hat(i) = gamma(i,1);
end

% State transition probability estimated
for j = 1:N
    for i = 1:N
        sigma_numerator = 0;
        for t = 1:T-1
            sigma_numerator = sigma_numerator + xi(i,j,t);
        end
        
        sigma_denumerator = 0;
        for t = 1:T-1
            sigma_denumerator = sigma_denumerator + gamma(i,t);
        end
        
        A_hat(i,j) = sigma_numerator/sigma_denumerator;
    end
end

% Emission probability estimated
for j = 1:N
    for k = 1:K
        sigma_numerator = 0;
        for t = 1:T
            if(O(t) == V(k))
                sigma_numerator = sigma_numerator + gamma(j,t);
            else
                sigma_numerator = sigma_numerator + 0;
            end
        end
        
        sigma_denumerator = 0;
        for t = 1:T
            sigma_denumerator = sigma_denumerator + gamma(j,t);
        end
        
        B_hat(j,k) = sigma_numerator/sigma_denumerator;
    end
end




