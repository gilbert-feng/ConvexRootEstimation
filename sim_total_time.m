clear; tic;  
%----------------------------
%% data generate process 
%----------------------------
rng('default')
rng(123);
rep = 10;

% total simulation setting
homo_seq = [1,0]; % homo = 1 homoskedasticity ;homo = 0 heteroskedasticity.
n_seq = [1,2]*1000; % number of sample
weight_num_seq = [3,5]; % weight matrix number
lambda_seq = [0.4, 0.6, -0.6]; % lambda value
p = 3; % number of covariates
beta = [1,-1,2]; % exogenous value
order = 0; % order of inverse approximation

for homo = homo_seq
    for n = n_seq
        % heter error
        if homo == 1
            h = ones(n,1);
        else
            h = ones(n,1)/2;
            n1 = ceil(n/3); 
            n2 = n - 2*n1;
            h((n1+1):(n1+n2)) = 1/10;
            h = n/sum(h)*h;
        end
        for weight_mat_num = weight_num_seq
           for lambda = lambda_seq
                  %% normal case
                gamma = flip(1:weight_mat_num);
                gamma = gamma/sum(gamma);
                %% zero case
%                 if weight_mat_num == 3
%                     gamma = ones(1,weight_mat_num);
%                     gamma = gamma/(weight_mat_num - 1);
%                     gamma(end) = 0;
%                 else
%                     gamma = ones(1,weight_mat_num);
%                     gamma = gamma/(weight_mat_num - 2);
%                     gamma((end - 1):end) = 0;
%                 end


                weight_total = cell(1, weight_mat_num);
                s_true = speye(n);
                for i = 1:weight_mat_num
                    % sparse
                    % weight_total{i} = make_neighborsw(normrnd(0,1,n,1),normrnd(0,1,n,1),i*2);
                    % not sparse
                    weight_total{i} = full(make_neighborsw(normrnd(0,1,n,1),normrnd(0,1,n,1),i*2));
                    %weight_total{i} = weight_spar(n, alpha);
                    %weight_total{i} = weight_block(n, (star_point(i)-(weight_matrix_param(i))):(star_point(i)+(weight_matrix_param(i))));
                    s_true = s_true - lambda * gamma(i)*weight_total{i};
                end
                true_value = [lambda, gamma, beta];
                true_value = true_value';

                X = normrnd(0,2,n,p);
                sige = 1; % std of error term
                beta_ind = 1:3; % impact measure
                %----------------------------
                %% Simulation
                %----------------------------
                % phi estimator
%                 impact_tensor = zeros(length(beta_ind), 3, rep);
%                 impact_se_tensor = zeros(length(beta_ind), 3, rep);
                phi_mat = zeros(length(true_value), rep);
                phi_se_mat = zeros(length(true_value), rep);
                % root estimator
%                 root_impact_tensor = zeros(length(beta_ind), 3, rep);
%                 root_impact_se_tensor = zeros(length(beta_ind), 3, rep);
                root_phi_mat = zeros(length(true_value), rep);
                root_phi_se_mat = zeros(length(true_value), rep);
                % time
                calt_phi = zeros(1,rep);
                calt_root = zeros(1,rep);
                
                
                % total_lambda = lambda1*w1-lambda2*w2-lambda3*w3;
                % sum((inv_s_true - inv_s_true.*(abs(inv_s_true)>1e-4)).^2, 'all')
                
                for mc = 1:rep
                    e = normrnd(0,sige,n,1);
                    V = e.*h;
                    y = s_true\(X * beta'+V);
                    
                    
                    %% Initial consistency estimator (2SLSE)
                    [delta, Wy] = est_initial(n,p,X,y,weight_mat_num,weight_total);
                    
                    %% Test for homo 
                %     V_n = y - X * delta((weight_mat_num + 1):end);
                %     for i = 1:weight_mat_num
                %         V_n = V_n - delta(i)*Wy(:,i);
                %     end
                %     Z_n = floor(h*2) + 1; % need to know the group structure
                %     % Z_n = floor(abs(V_n)) + 1; % based on estimated error
                %     homo = error_test(V_n, Z_n, 0.05);
                    
                    
                    %% Estimation
                    tic;
                    if homo == 1 % QMLE
                        [phi_est,phi_sd,Wn,phi_SE] = qmle(y,Wy,X,delta(1:weight_mat_num),weight_total);
                    else % GMM
                        [phi_est,phi_sd,Wn,phi_SE] = gmm(y,Wy,X,delta,weight_total);
                    end
                    calt_phi(mc) = toc;
                    % impact measure
%                     [impact_phi,impact_sd_mat] = impact(X,phi_est,phi_SE,Wn,beta_ind,order,weight_total);
                
                    tic;
                    %% root estimator
                    [root,root_sd,Wn,root_SE] = root_est(y,X,homo,order,delta,weight_total);
                    calt_root(mc) = toc;
                    % impact measure
%                     [impact_root,impact_root_sd_mat] = impact(X,root,root_SE,Wn,beta_ind,order,weight_total);
                
                    %% save results
                    % phi estimator
                    phi_mat(:,mc) = phi_est;
                    phi_se_mat(:,mc) = phi_sd;
%                     impact_tensor(:,:,mc) = impact_phi;
%                     impact_se_tensor(:,:,mc) = impact_sd_mat;
                    % root estimator
                    root_phi_mat(:,mc) = root;
                    root_phi_se_mat(:,mc) = root_sd;
%                     root_impact_tensor(:,:,mc) = impact_root;
%                     root_impact_se_tensor(:,:,mc) = impact_root_sd_mat;
                
                    %% simulation monitor
                    show=['Simulation process updated for some MC ',num2str(100*mc/rep), '%. "',num2str(homo),'-',num2str(n),'-',num2str(weight_mat_num),'-',num2str(lambda),'" case complete!'];   
                    disp(show)
                end
                
                %% phi result
                phi_bias = mean(phi_mat,2)-true_value;
                phi_demean = phi_mat - mean(phi_mat,2);
                phi_rmse = sqrt(mean((phi_mat-true_value).^2,2));
                phi_std = sqrt(mean(phi_demean.^2,2));
                phi_result = [phi_bias,phi_rmse,phi_std,mean(phi_se_mat,2)];
                % root result
                root_phi_bias = mean(root_phi_mat,2)-true_value;
                root_phi_demean = root_phi_mat - mean(root_phi_mat,2);
                root_phi_rmse = sqrt(mean((root_phi_mat-true_value).^2,2));
                root_phi_std = sqrt(mean(root_phi_demean.^2,2));
                root_phi_result = [root_phi_bias,root_phi_rmse,root_phi_std,mean(root_phi_se_mat,2)];
                
                %% impact result
%                 inv_s_true = s_true\speye(n);
%                 true_impact_mat = zeros(length(beta_ind), 3);
%                 temp_counter = 1;
%                 for i = (beta_ind + weight_mat_num + 1)
%                     true_impact_mat(temp_counter,1) = 1/n*trace(inv_s_true*true_value(i));
%                     true_impact_mat(temp_counter,2) = 1/n*sum(inv_s_true, 'all')*true_value(i);
%                     true_impact_mat(temp_counter,3) = true_impact_mat(temp_counter,2) - true_impact_mat(temp_counter,1);
%                     temp_counter = temp_counter + 1;
%                 end
                
%                 % phi result
%                 impact_bias = mean(impact_tensor, 3) - true_impact_mat;
%                 impact_demean = impact_tensor - mean(impact_tensor, 3);
%                 impact_rmse = sqrt(mean((impact_tensor - true_impact_mat).^2,3));
%                 impact_std = sqrt(mean(impact_demean.^2,3));
%                 impact_result = [vec(impact_bias),vec(impact_rmse),vec(impact_std),vec(mean(impact_se_tensor,3))];
%                 % root result
%                 root_impact_bias = mean(root_impact_tensor, 3) - true_impact_mat;
%                 root_impact_demean = root_impact_tensor - mean(root_impact_tensor, 3);
%                 root_impact_rmse = sqrt(mean((root_impact_tensor - true_impact_mat).^2,3));
%                 root_impact_std = sqrt(mean(root_impact_demean.^2,3));
%                 root_impact_result = [vec(root_impact_bias),vec(root_impact_rmse),vec(root_impact_std),vec(mean(root_impact_se_tensor,3))];
%                 
                %% time result
                time_result = [calt_phi;calt_root];
                time_result = [mean(time_result,2),std(time_result,0,2)];
                
                %----------------------------
                %% Save files 
                %----------------------------
                % phi result
                if homo == 1
                    prefix = 'simulation\homo\';
                else
                    prefix = 'simulation\hete\';
                end
                
                % Original results
                phi_mat_table = table(phi_mat);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\phi est - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(phi_mat_table,filename); 
                phi_mat_table = table(phi_se_mat);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\phi est sd - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(phi_mat_table,filename); 
                % RMSE
                row_names = {'bias','rmse','std','se'};
                phi_table = table(phi_result', 'RowNames', row_names);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\phi est rmse - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(phi_table,filename, 'WriteRowNames', true); 
%                 impact_table = table(impact_result', 'RowNames', row_names);
%                 filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\phi est impact - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
%                 writetable(impact_table,filename, 'WriteRowNames', true); 
                
                % root result
                % Original results
                root_phi_mat_table = table(root_phi_mat);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\root phi est - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(root_phi_mat_table,filename); 
                root_phi_mat_table = table(root_phi_se_mat);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\root phi est sd - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(root_phi_mat_table,filename); 
                % RMSE
                row_names = {'bias','rmse','std','se'};
                root_phi_table = table(root_phi_result', 'RowNames', row_names);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\root phi est rmse - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(root_phi_table,filename, 'WriteRowNames', true); 
%                 root_impact_table = table(root_impact_result', 'RowNames', row_names);
%                 filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\root phi est impact - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
%                 writetable(root_impact_table,filename, 'WriteRowNames', true); 
                row_names = {'mean', 'std'};
                % time result
                time_table = table(time_result', 'RowNames', row_names);
                filename = [prefix, 'n = ',num2str(n), '\lambda = ',num2str(lambda), '\Time - weight_mat_num = ',num2str(weight_mat_num),'.csv'];
                writetable(time_table,filename, 'WriteRowNames', true); 
           end
        end 
    end
end


