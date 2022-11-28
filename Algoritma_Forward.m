% INISIALISASI SPEK MODEL
syms A B;
S = [A B]; % misal S = {s1, s2} = {A, B}

syms x y z; % Berdasarkan model sebelumnya alfa = x, beta = y, gamma = z. Diganti supaya tidak rancu
V = [x y z]; % misal V = {v1, v2, v3} = {x, y, z}

pi = [0.8 0.2]; % initial probability
A = [0.7 0.3; 0.4 0.6]; % state transition probability
B = [0.2 0.4 0.4;0.5 0.4 0.1]; % emission probability

O = [x z x]; % observable sequence
%Q = zeros(1,length(O)); % hidden state sequence

N = length(S); % jumlah state
T = length(O); % jumlah observasi

alfa = zeros(N,T); % forward variable
P_O_lambda = 0; % probabability of observation sequence based on model (lambda)

% START ALGORITHM

% 1. Initialization
for i = 1:N
    alfa(i,1) = pi(i)*B(i,find(V == O(1)));
end

% 2. Induction
for t = 1:(T-1)
    for j = 1:N
        sigma_alfa_a = 0;
        for i = 1:N
            sigma_alfa_a = sigma_alfa_a + alfa(i,t)*A(i,j);
        end
        alfa(j,t+1) = sigma_alfa_a*B(j,find(V == O(t+1)));
    end
end

% 3. Termination
for i = 1:N
    P_O_lambda = P_O_lambda + alfa(i,T);
end


