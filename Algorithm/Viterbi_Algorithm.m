% INITIALIZE HMM MODEL SPECIFICATION
syms A B;
S = [A B]; % hidden states

syms alfa beta gamma;
V = [alfa beta gamma]; % observable states

pi = [0.8 0.2]; % initial probability
A = [0.7 0.3; 0.4 0.6]; % state transition probability
B = [0.2 0.4 0.4;0.5 0.4 0.1]; % emission probability

O = [alfa gamma alfa]; % observable sequence

N = length(S); % number of distinct states
T = length(O); % number of data in observable sequence

% START ALGORITHM

% Calculate weight
P = zeros(N^2,T);
for k = 1:T
    for j = 1:N
        for i = 1:N
            if k == 1
                P(i+N*(j-1),k) = B(i,find(V == O(k)))*pi(i); 
            else
                P(i+N*(j-1),k) = B(i,find(V == O(k)))*A(j,i);
            end
        end
    end
end

% Calculate state value
W = zeros(N,T);
X = zeros(1,N);
for k = 1:T
    for j = 1:N
        if k == 1
            W(j,k) = P(j,k);
        else
            for i = 1:N
                if i == 1
                    X(i) = P(j,k)*W(i,k-1);
                else
                    X(i) = P(i+N*(j-1),k)*W(i,k-1);
                end
                W(j,k) = max(X);
            end
        end
    end
end

% Determine hidden states sequence based on nilai state value
q = zeros(1,length(O)); % hidden states sequence
for k = 1:T
    q(k) = find(W(:,k) == max(W(:,k)));
    Q(k) = S(q(k)); % hidden state sequence in symbol
end



