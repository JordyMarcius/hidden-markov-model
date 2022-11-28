%% HMM MODEL SPECIFICATION INITIALIZTION

fileName = 'data_1.xlsx';
sheetName = 'sensorDepan';
data_sensorDepan = readtable(fileName, 'Sheet', sheetName);
pitch = data_sensorDepan(:, 16);
time = data_sensorDepan(:, 2);
data_pitch = table2array(pitch); % convert table to array
data_time = table2array(time);

data_pitch = abs(data_pitch(1:2212)); % absolute pitch data for testing model
abs_pitch_1 = abs(data_pitch(1:1000)); % absolute pitch data for train model

epoch = 1;

% INITIALIZATION (MODEL SPECIFICATION)
S = [0 1]; % state (0 -> NORMAL, 1 -> ABNORMAL)
V = [0 1]; % observable state (0 -> pitch <= 15, 1 -> pitch > 15)

pi = [0.5 0.5]; % initial probability
A = [0.5 0.5; 0.5 0.5]; % state transition probability
B = [0.6 0.4; 0.4 0.6]; % emission probability

%pi_hat = zeros(epoch_end,length(pi));
%A_hat = zeros(length(A(:,1)),length(A),epoch_end);
%B_hat = zeros(length(B(:,1)),length(B),epoch_end);

threshold = 15;

O = zeros(length(abs_pitch_1),1); % observable state sequence
for i = 1:length(abs_pitch_1)
    if abs_pitch_1(i) > threshold
        O(i) = 1;
    else
        O(i) = 0;
    end
end

O_test = zeros(length(data_pitch),1);
for i = 1:length(data_pitch)
    if data_pitch(i) > threshold
        O_test(i) = 1;
    else
        O_test(i) = 0;
    end
end

N = length(S); % number of distinct hidden states
K = length(V); % number of distinct observable states
T = length(O); % number of observable data in sequence
T_test = length(O_test);

flag_stop = 0;

%% TRAINING MODEL ALGORITHM
while flag_stop == 0
    alfa = zeros(N,T); % forward variable
    beta = zeros(N,T); % backward variable
    gamma = zeros(N,T); % probability of being state i in time t
    xi = zeros(N^2,T-1); % probability of being state i in time t and state j in time t+1
    P_O_lambda = 0; % probabability of observation sequence based on model (lambda)
    
    % FORWARD ALGORITHM TO CALCULATE ALFA VARIABLE
    % 1. Initialization
    for i = 1:N
        alfa(i,1) = pi(i)*B(i, V(:)==O(1));
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
                sigma_a_b_beta = sigma_a_b_beta + A(i,j)*B(j,V(:)==O(t+1))*beta(j,t+1);
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
                xi(i,j,t) = (alfa(i,t)*A(i,j)*B(j,V(:)==O(t+1))*beta(j,t+1))/P_O_lambda;
            end
        end
    end

    % Initial probability estimated
    for i = 1:N
        pi_hat(epoch,i) = gamma(i,1);
        pi(i) = pi_hat(epoch,i); % update
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

            A_hat(i,j,epoch) = sigma_numerator/sigma_denumerator;
            A(i,j) = A_hat(i,j,epoch); % update
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

            B_hat(j,k,epoch) = sigma_numerator/sigma_denumerator;
            B(j,k) = B_hat(j,k,epoch); % update
        end
    end
    
    if(epoch == 1)
        diff_pi1 = abs(pi_hat(epoch,1)-0);
        diff_pi2 = abs(pi_hat(epoch,2)-0);
        diff_a11 = abs(A_hat(1,1,epoch)-0);
        diff_a12 = abs(A_hat(1,2,epoch)-0);
        diff_a21 = abs(A_hat(2,1,epoch)-0);
        diff_a22 = abs(A_hat(2,2,epoch)-0);
        diff_b11 = abs(B_hat(1,1,epoch)-0);
        diff_b12 = abs(B_hat(1,2,epoch)-0);
        diff_b21 = abs(B_hat(2,1,epoch)-0);
        diff_b22 = abs(B_hat(2,2,epoch)-0);
    else
        diff_pi1 = abs(pi_hat(epoch,1) - pi_hat(epoch-1,1));
        diff_pi2 = abs(pi_hat(epoch,2) - pi_hat(epoch-1,2));
        diff_a11 = abs(A_hat(1,1,epoch) - A_hat(1,1,epoch-1));
        diff_a12 = abs(A_hat(1,2,epoch) - A_hat(1,2,epoch-1));
        diff_a21 = abs(A_hat(2,1,epoch) - A_hat(2,1,epoch-1));
        diff_a22 = abs(A_hat(2,2,epoch) - A_hat(2,2,epoch-1));
        diff_b11 = abs(B_hat(1,1,epoch) - B_hat(1,1,epoch-1));
        diff_b12 = abs(B_hat(1,2,epoch) - B_hat(1,2,epoch-1));
        diff_b21 = abs(B_hat(2,1,epoch) - B_hat(2,1,epoch-1));
        diff_b22 = abs(B_hat(2,2,epoch) - B_hat(2,2,epoch-1));
    end
    
    if(diff_pi1<0.001 && diff_pi2<0.001 && diff_a11<0.001 && diff_a12<0.001 && diff_a21<0.001 && diff_a22<0.001 && diff_b11<0.001 && diff_b12<0.001 && diff_b21<0.001 && diff_b22<0.001)
        flag_stop = 1;
    end
    
    epoch = epoch + 1;
end


%% VITERBI ALGORTIHM
% Calculate weight
P = zeros(N^2,T_test);
for k = 1:T_test
    for j = 1:N
        for i = 1:N
            if k == 1
                P(i+N*(j-1),k) = log(B(i,V(:)==O_test(k))) + log((pi(i))); 
            else
                P(i+N*(j-1),k) = log(B(i,V(:)==O_test(k))) + log(A(j,i));
            end
        end
    end
end

% Calculate states value
W = zeros(N,T_test);
X = zeros(1,N);
for k = 1:T_test
    for j = 1:N
        if k == 1
            W(j,k) = P(j,k);
        else
            for i = 1:N
                if i == 1
                    X(i) = P(j,k) + W(i,k-1);
                else
                    X(i) = P(i+N*(j-1),k) + W(i,k-1);
                end
                W(j,k) = max(X);
            end
        end
    end
end

% Determine hidden states sequence based on maximum states value in time t
q = zeros(length(O_test),1); % hidden states sequence
Q = zeros(length(O_test),1); % hidden states sequence
for k = 1:T_test
    q(k) = find(W(:,k) == max(W(:,k)));
    Q(k) = S(q(k));
end


%% PLOT HASIL
fig_1 = figure('Name','TEST 1');
tabgroup_1 = uitabgroup(fig_1);

% ---- Tab 1 ----
tab_1 = uitab(tabgroup_1, 'Title', 'Line Plot');
ax_1 = axes('Parent', tab_1);
plot(ax_1, data_time(1:2212), abs(data_pitch(1:2212)));
title('Plot Pitch Driving Phase-1');
xlabel('time');
ylabel('pitch');

% ---- Tab 2 ----
tab_2 = uitab(tabgroup_1, 'Title', 'Scatter Plot');
ax_2 = axes('Parent', tab_2);
scatter(ax_2, abs(data_pitch(1:2212)), abs(data_pitch(1:2212)));
title('Scatter Plot Driving Phase-1');
xlabel('pitch');
ylabel('pitch');

% ---- Tab 3 ----
tab_3 = uitab(tabgroup_1, 'Title', '1:200');
ax_3 = axes('Parent', tab_3);
plot(ax_3, data_time(1:length(abs_pitch_1)), abs_pitch_1);
title('Plot Driving Phase-1 data 1 - 200');
xlabel('time');
ylabel('pitch');

% ---- Tab 4 ----
tab_4 = uitab(tabgroup_1, 'Title', 'Pi');
ax_4 = axes('Parent', tab_4);
plot(ax_4, pi_hat);
legend('pi_0','pi_1')
title('Plot Training Pi (Initial Probability)');
xlabel('epoch');
ylabel('pi');

% ---- Tab 5 ----
tab_5 = uitab(tabgroup_1, 'Title', 'A');
ax_5 = axes('Parent', tab_5);
A_plot11 = zeros(epoch-1,1);
A_plot12 = zeros(epoch-1,1);
A_plot21 = zeros(epoch-1,1);
A_plot22 = zeros(epoch-1,1);
for i = 1:epoch-1
    A_plot11(i) = A_hat(1,1,i);
    A_plot12(i) = A_hat(1,2,i);
    A_plot21(i) = A_hat(2,1,i);
    A_plot22(i) = A_hat(2,2,i);
end
plot(ax_5, A_plot11);
hold on
plot(ax_5, A_plot12);
hold on
plot(ax_5, A_plot21);
hold on
plot(ax_5, A_plot22);
legend('a_0_0','a_0_1','a_1_0','a_1_1');
title('Plot Training A (State Transistion Probability)');
xlabel('epoch');
ylabel('A');

% ---- Tab 6 ----
tab_6 = uitab(tabgroup_1, 'Title', 'B');
ax_6 = axes('Parent', tab_6);
B_plot11 = zeros(epoch-1,1);
B_plot12 = zeros(epoch-1,1);
B_plot21 = zeros(epoch-1,1);
B_plot22 = zeros(epoch-1,1);
for i = 1:epoch-1
    B_plot11(i) = B_hat(1,1,i);
    B_plot12(i) = B_hat(1,2,i);
    B_plot21(i) = B_hat(2,1,i);
    B_plot22(i) = B_hat(2,2,i);
end
plot(ax_6, B_plot11);
hold on
plot(ax_6, B_plot12);
hold on
plot(ax_6, B_plot21);
hold on
plot(ax_6, B_plot22);
legend('b_0(0)','b_0(1)','b_1(0)','b_1(1)');
title('Plot Training B (Emission Probability)');
xlabel('epoch');
ylabel('B');

% ---- Tab 7 ----
tab_7 = uitab(tabgroup_1, 'Title', 'state');
ax_7 = axes('Parent', tab_7);
b = bar(ax_7, Q*max(data_pitch));
b.FaceColor = "#ffffb3";
b.EdgeColor = "#ffffb3";
hold on
plot(ax_7, data_pitch);
title('Plot Hidden State Sequence');
xlabel('data ke-');
ylabel('');








