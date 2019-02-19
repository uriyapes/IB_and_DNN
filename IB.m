clc;
clear all;
close all;

x_size_vec = 200;
max_barriers = 4;
min_barriers = 4;
barriers_vec = min_barriers:max_barriers;
shuffle_prob_flag = true;

t_size_min = 2;
t_size_max = 5;
t_size_vec = [t_size_min:t_size_max 9];
% t_size_vec=7;
IB_results_st = struct('Ixt', [], 'Iyt', [], 'beta', [], 'Pt_x', [{}]);
results_st_arr = struct('seed', [], 'M', [], 't_size', [], 'max_barriers', []...
                             ,'quant_result', IB_results_st, 'IB_result', IB_results_st);
iter = 1;

rng('shuffle', 'twister')
s = rng;
% s.Seed = 103237715; %31527649 - 2Beta shows bad value 31527649-new version working % 89253521-not working for 6 barriers
rng(s.Seed);%10608982 %s.Seed 

plot_color_index = {'k','b','r','g','y','c',[.5 .6 .7],[.8 .2 .6]};
plot_marker_index = {'o','+','*','s','d','^','v','>','p','h'};
for barriers = barriers_vec
   for M = x_size_vec%max(t_size,barriers+1)+1:4:16
        %%
        %Create a fine DMC based on barrier model with M outputs.
        [Pyx, Px_y_tr, Py, x_vec] = biKbarrier2Dmc(M, barriers, [0.5;0.5], shuffle_prob_flag);
        %   Px_y_tr is a 2-by-M matrix, where: P(j,m) = Pr( X=m | Y=j ). Notice:
        %   Px_y_tr here is transposed
        for t_size = t_size_vec%t_size_min:min(barriers+1, t_size_max)
            %%
            % Save relevant iteration run configuration to the struct results_st_arr
            fprintf('barriers=%d  t_size=%d   X size=%d\n', barriers, t_size, M);
            results_st_arr = setfield(results_st_arr, 'M', {iter}, M);
            results_st_arr = setfield(results_st_arr, 't_size', {iter}, t_size);
            results_st_arr = setfield(results_st_arr, 'max_barriers', {iter}, max_barriers);

            results_st_arr.seed(iter) = s.Seed;
            fprintf('seed is %d\n', results_st_arr.seed(iter));
            
            %perform quantization
            [Q, mi] = quantBiDmcMulti(Pyx, t_size);
            fprintf('number of quantization solutions are %d\n', size(Q,2));
            Iyz = max(mi);
            assert(all(ismembertol(Iyz, mi))); %all optimal quantiziers should have the same Iyz
            [Pz, Py_x, Py_z, Pz_x, Ixz] = chooseBestQuant(Q, Pyx, Py);
            figure(iter+1)
            [row,col] = find(Pz_x);
            stem(x_vec, row)
            title('q Vs. X where P(T=q|x) = 1')
            
            real_beta = findBetaNumerically(Pz, Py_x, Py_z, Pz_x, 1e-6)
%             real_beta = findBeta(Pz, Py_x, Py_z, Pz_x, 1e-5)
            if isempty(real_beta) % no solution
                fprintf('NO BETA WAS FOUND\n');
                real_beta = 1900;
            end
            results_st_arr.quant_result(1).beta(iter) = real_beta;
            results_st_arr.quant_result.Ixt(iter) = Ixz;
            results_st_arr.quant_result.Iyt(iter) = Iyz;
            
            
            % We must restrict beta otherwise the exp(-beta*...) in iIB will result in
            % 0 and
            max_beta_value = 1800;
            if real_beta > max_beta_value
                beta = max_beta_value;
            else
                beta = real_beta;
            end
            
            beta_vec_size = 40;
            if(iter==1 || beta==max_beta_value)
                beta_vec = linspace(1, 1*beta, beta_vec_size);%[logspace(log10(10), log10(beta), 15) beta, 1.3*beta];
            else
                half_old_beta_vec = beta_vec(1:2:end);%beta_vec(floor(beta_vec_size/2):end);
                beta_vec =[half_old_beta_vec linspace(beta_vec(end-1), 1*beta, beta_vec_size-length(half_old_beta_vec))] ;
            end
            [Pt_x , Ixt , Iyt , Ptx , last, Py_t, convergence_flag_vec] = iterativeiIB(Pyx' , beta_vec , t_size , .5 , .5 , 10^-12 , 'log' , 20);
            results_st_arr.IB_result(1).beta(:, iter) = beta_vec';
            results_st_arr.IB_result.Ixt(:, iter) = Ixt';
            results_st_arr.IB_result.Iyt(:, iter) = Iyt';
            results_st_arr.IB_result.Pt_x(iter) = {Pt_x};
            results_st_arr.IB_result.convergence_flag(:,iter) = convergence_flag_vec;
            
            %%
            figure(1)
            hold on
            scatter(results_st_arr.IB_result.Ixt(:,iter), results_st_arr.IB_result.Iyt(:,iter), 'MarkerEdgeColor', plot_color_index{iter},'Marker', plot_marker_index{iter});
            ylabel('I(Y;T) [bits]');
            xlabel_txt = sprintf('I(X;T) [bits]\nSeed=%d',s.Seed);
            xlabel(xlabel_txt);
%             TODO: add relative errors, add more equations for the beta
%             solver to be sure the solution is on the graph and add noise
            delta_Ixt = 100*(results_st_arr.quant_result.Ixt(iter) - results_st_arr.IB_result.Ixt(end,iter))/results_st_arr.quant_result.Ixt(iter);
            delta_Iyt = 100*(results_st_arr.quant_result.Iyt(iter) - results_st_arr.IB_result.Iyt(end,iter))/results_st_arr.quant_result.Iyt(iter);
            title_txt = sprintf('IB curve for %d barriers model with |T|=%d  |X|=%d \n BETA=~%d relative error Ixt=%d%% Iyt=%d%%', barriers, t_size, M, beta_vec(end), delta_Ixt, delta_Iyt);
            title(title_txt);
            hold on;
            scatter(results_st_arr.quant_result.Ixt(iter), results_st_arr.quant_result.Iyt(iter),500, 'MarkerEdgeColor', plot_color_index{iter}, 'Marker','X');
            legend_txt{2*iter-1} = sprintf('T=%d', t_size);
            legend_txt{2*iter} = sprintf('Opt. quant T=%d', t_size);
            %add beta values as text near the point in the graph.
%             c = cellstr(num2str(results_st_arr.IB_result.beta(:,iter)));
%             text(results_st_arr.IB_result.Ixt(:,iter), results_st_arr.IB_result.Iyt(:,iter)-0.001,c)
            iter = iter + 1;                      
        end
    end    
end
legend(legend_txt, 'Location', 'southeast');
datetime.setDefaultFormats('default','dd-MM hh_mm_ss')
filename = "results\"+string(datetime)+".mat";
save(filename)
% 
% Pt_x
% Pz_x
% fprintf('## Optimal Quantizer Vs IB with Beta=%d (real Beta=%d) ##\n', beta, real_beta);
% fprintf('Optimal Quantizer: Ixz=%d   Iyz=%d\n', Ixz, Iyz);
% fprintf('############## IB: Ixt=%d   Iyt=%d\n', Ixt, Iyt);

function [Pz_best, Py_x, Py_z_best, Pz_x_best, Ixz] = chooseBestQuant(Q, Pyx, Py)
    best_index = 0;
%     assert(size(Q,2) == 1);
    Ixz_best = Inf;
    for i=1:size(Q,2)
        [Pz, Py_x, Py_z, Ixz] = calcQuantProb(Q{i}, Pyx, Py);
        fprintf('quantizer number %d Ixz = %d\n', i, Ixz);
        if Ixz < Ixz_best
            Pz_best = Pz;
            Py_z_best = Py_z;
            Pz_x_best = Q{i};
            Ixz_best = Ixz;
            best_index = i;
        end
    end
    fprintf('quantizer number %d is the best with Ixz = %d\n', best_index, Ixz_best);
end

function [Pz, Py_x, Py_z, Ixz] = calcQuantProb(Pz_x, Pyx, Py)
    x_size = size(Pyx,2);
    t_size = size(Pz_x,1);
    Px_y_tr = Pyx ./ repmat(Py, 1, x_size);
    Px = sum(Pyx, 1);% p(x) = SigmaY(p(x|y)*p(y))
    Py_x = Pyx ./ Px;

    % Pz_x - P(z|x) is a KxM matrix, each row represent a different
    % z (quantization) value and each colum represent a different x value. 
    % P(j,m) = Pr(Z=j | x=m)
    Pz = sum(Pz_x .* repmat(Px, t_size, 1), 2);
    assert(ismembertol(sum(Pz),1)); % Check that Pz satasfies Eq.1 in IB paper.
    Pzx = Pz_x .* repmat(Px, t_size, 1);
    assert(ismembertol(sum(sum(Pzx)), 1));
    Ixz_mat = Pzx .* log2(Pz_x./repmat(Pz, 1, x_size));
    Ixz_mat(isnan(Ixz_mat)) = 0;
    Ixz = sum(sum(Ixz_mat));
    Pz_y = Pz_x * Px_y_tr';

    Pzy2 = Pz_y .* repmat(Py',t_size,1);
    Pzy = (Pyx * Pz_x')';
    assert(all(all(ismembertol(Pzy ,Pzy2))));
    assert(all(ismembertol(sum(Pzy,1), Py))); 
    Py_z = (Pzy ./ repmat(Pz,1,size(Pzy,2)))';
    
end

function beta = findBetaNumerically(Pz, Py_x, Py_z, Pz_x, epsilon)
%     B = 1:1:5e4;
    B = [1:1:2e3 2.001e3:100:2e4 2.001e4:1e4:5e5];
    x_size = size(Py_x, 2);
    Zpart = [];
    for x=1:x_size
        temp = 0;
        for z=1:length(Pz)
            temp = temp + Pz(z).*exp(-B .* KLDiv(Py_x(:,x), Py_z(:,z)));
        end
        Zpart = [Zpart; temp];
    end
    eqn = [];
    for x=1:x_size
        for z=1:length(Pz)
            eq = abs(Pz(z).*exp(-B .* KLDiv(Py_x(:,x), Py_z(:,z))) ./ Zpart(x,:) - Pz_x(z,x)) < epsilon;
            eqn = [eqn; eq];
        end
    end
    beta = find(all(eqn,1),1);
    
end


function beta = findBeta(Pz, Py_x, Py_z, Pz_x, epsilon)
    % Try to find beta value that will fulfill Eq.16 for every set P(z|x)
    syms B
    assume(B>0)
%     cond = in(B, 'integer') not helping at all
%     assume(cond)
    x_size = size(Py_x, 2);
    Zpart = [];
    for x=1:x_size
        temp = 0;
        for z=1:length(Pz)
            temp = temp + Pz(z)*exp(-B * KLDiv(Py_x(:,x), Py_z(:,z)));
        end
        Zpart = [Zpart, temp];
    end
    eqn = [];
    for x=1:x_size
        for z=1:length(Pz)
            eq = abs(Pz(z)*exp(-B * KLDiv(Py_x(:,x), Py_z(:,z)))/Zpart(x) - Pz_x(z,x)) < epsilon;
            eqn = [eqn, eq];
        end
    end
%     Zpart = Pz(1)*exp(-B * KLDiv(Py_x(:,1), Py_z(:,1))) + Pz(2)*exp(-B * KLDiv(Py_x(:,1), Py_z(:,2)));
%     eqn1 = abs(Pz(1)*exp(-B * KLDiv(Py_x(:,1), Py_z(:,1)))/Zpart - Pz_x(1,1)) < 1e-5;
%     eqn2 = abs(Pz(2)*exp(-B * KLDiv(Py_x(:,1), Py_z(:,2)))/Zpart - Pz_x(2,1)) < 1e-5;
%     eqn = [eqn1, eqn2];
    
    beta = solve(eqn, B, 'IgnoreAnalyticConstraints', true);
    beta = double(beta);
end


function dist = KLDiv(P,Q)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  Input:
%   P - prob. colum vector
%   Q - Prob. colum vector
    if nargin < 2
        P = [0.25 0.25 0.25 0.25]';
        Q = [0.25 0.25 0.35 0.15]';
    end
    
    assert(isequal(size(P), size(Q)));
    assert(ismembertol(sum(sum(P)),1));
    assert(ismembertol(sum(sum(Q)),1));
    
    temp = P .* log2(P./Q);
    temp(isnan(temp))=0; % resolving the case when P(i)==0
    dist = sum(temp,1);
end






