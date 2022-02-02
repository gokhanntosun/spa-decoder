clear;
tic
% Generator matrix
G = [
    0 1 1 0 1 1 0 0 0 0;
    0 1 0 1 1 0 1 0 0 0;
    1 0 1 0 1 0 0 1 0 0;
    1 0 0 1 1 0 0 0 1 0;
    1 1 1 1 1 0 0 0 0 1;
    ];

% Parity-check matrix
H = [
    1 1 1 1 0 0 0 0 0 0;
    1 0 0 0 1 1 1 0 0 0;
    0 1 0 0 1 0 0 1 1 0;
    0 0 1 0 0 1 0 1 0 1;
    0 0 0 1 0 0 1 0 1 1;
    ];

% Row weight
wr = sum(H(1,:));

% Column weight
wc = sum(H(:,1));

% Number of c-nodes
num_c = size(H, 1);

% Number of v-nodes
num_v = size(H, 2);

% Maximum number of BP iterations
max_iter = 10;

% Batch size
bs = 5*1e4;

% SNR in Db
snr_db = 0:2:10;
vars = 1./(10.^(snr_db./10));

% BER vector
ber = zeros(1, bs);

% SNR vector
snr_ber = zeros(1, length(snr_db));
u_ber = zeros(1, length(snr_db));

for m = 1:length(snr_db)
    % Random information bit sequence
    u = randi([0 1], [bs, 5]);

    % Encode bits
    v = mod(u * G, 2);

    % Convert to +1 -1
    v = 1 - 2 .* v;
    
    % Noise variance 
    var = vars(m);
    
    % Add AWGN
    r = v + sqrt(var).*randn(size(v));
    ru = (1 - 2 .* u) + sqrt(var).*randn(size(u));
    
    % Begin decoding
    for w = 1:bs
        % Received codeword
        y = r(w,:);
        
        % Initialize v-nodes
        for i = 1:num_v
            vnodes(i) = struct('q0', zeros(1, wc), ...
                'q1', zeros(1, wc), ...
                'Q0', 0, ...
                'Q1', 0, ...
                'n', find(H(:, i)==1)'...
                ); %#ok<SAGROW>
        end
        
        % Initialize c-nodes
        for i = 1:num_c
            cnodes(i) = struct('r0', zeros(1, wr), ...
                'r1', zeros(1, wr), ...
                'n', find(H(i, :)==1)...
                ); %#ok<SAGROW>
        end
        
        % Compute syndrome
        s = mod(y * H', 2);
        
        % Begin message passing
        % Calculate LLRs
        P0 = 1./(1 + exp((-2.*y)./var));
        P1 = 1./(1 + exp((2.*y)./var));
        
        % Initialize q_ij(0) and q_ij(1) values
        for i = 1:num_v
            vnodes(i).q0 = P0(i) * ones(1, wc);
            vnodes(i).q1 = P1(i) * ones(1, wc);
        end
        
        % Begin iterations
        iter = 0;
        while (sum(s) ~= 0 && iter < max_iter)
            % Increase iteration count
            iter = iter + 1;
            
            % Update c-nodes
            for j = 1:num_c
                % Number of neigbor v-nodes
                nn = length(cnodes(j).n);
                % Exclude one neigbor v-node at a time
                exc = 1;
                % Iterate over neigbor v-nodes
                for i = 1:nn
                    % Product
                    prod = 1;
                    for k = 1:nn
                        % Current v-node
                        cv = vnodes(cnodes(j).n(k));
                        % Calculate product for r
                        prod = (k == exc) * prod + ...
                            (k ~= exc) * prod * (1 - 2 * cv.q1(find(cv.n == j)));
                    end
                    % Update r values
                    cnodes(j).r0(i) = 0.5 * (1 + prod);
                    cnodes(j).r1(i) = 1 - cnodes(j).r0(i);
                    % Update excluded node
                    exc = exc + 1;
                end
            end
            
            % Update v-nodes
            for i = 1:num_v
                % Number of neigbor c-nodes
                nn = length(vnodes(i).n);
                
                % Exclude one neigbor node at a time
                exc = 1;
                
                % Iterate over neigbor c-nodes
                for j = 1:nn
                    % Product
                    prod0 = 1;
                    prod1 = 1;
                    for k = 1:nn
                        % Current c-node
                        cc = cnodes(vnodes(i).n(j));
                        
                        % Product for q0
                        prod0 = (k == exc) * prod0 + ...
                            (k ~= exc) * cc.r0(find(cc.n == i)) * prod0;
                        
                        % Product for q1
                        prod1 = (k == exc) * prod1 + ...
                            (k ~= exc) * cc.r1(find(cc.n == i)) * prod1;
                    end
                    % Solve for K_ij
                    K = 1/(P0(i) * prod0 + P1(i) * prod1);
                    % Update q values
                    vnodes(i).q0(j) = K * P0(i) * prod0;
                    vnodes(i).q1(j) = K * P1(i) * prod1;
                    % Update excluded node
                    exc = exc + 1;
                end
            end
            
            % Compute Q_ij(0) and Q_ij(1)
            for i = 1:num_v
                % Number of neigbor c-nodes
                nn = length(vnodes(i).n);
                
                % Product
                prod0 = 1;
                prod1 = 1;
                
                % Iterate over neighbor c-nodes
                for j = 1:nn
                    % Current c-node
                    cc = cnodes(vnodes(i).n(j));
                    
                    % Product for Q0
                    prod0 = prod0 * cc.r0(find(cc.n == i));
                    % Product for Q1
                    prod1 = prod1 * cc.r1(find(cc.n == i));
                end
                % Solve for K_i
                K = 1/(P0(i) * prod0 + P1(i) * prod1);
                % Update Q0 and Q1
                vnodes(i).Q0 = K * P0(i) * prod0;
                vnodes(i).Q1 = K * P1(i) * prod1;
            end
            
            % Compute estimated codeword
            c = double([vnodes.Q1] > [vnodes.Q0]);
            
            % Compute syndrome
            s = mod(c * H', 2);
            
            % Encoding
            c = 1 - 2 * c;
        end
        
        ber(1, w) = nnz(v(w, :) - c);
    end
    
    ru(ru >= 0) = 1;
    ru(ru < 0) = -1;

    u_ber(1, m) = sum(nnz(ru - (1 - 2 .* u)))/(bs * size(G, 1));
    snr_ber(m) = sum(ber)/(bs * size(G,2));
end

figure;
semilogy(snr_db, snr_ber, '*-b', 'LineWidth', 2); hold on;
semilogy(snr_db, u_ber, '*-r', 'LineWidth', 2);
grid on; axis square;
xlabel('SNR (dB)');ylabel('BER');
legend('LDPC | (10,5)', 'Uncoded');ylim([1e-6 10]);
title('BPSK | AWGN Channel');
toc
