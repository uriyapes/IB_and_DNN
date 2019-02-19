function [Pt_x , Ixt , Iyt , Ptx, Py_t, convergence_flag] = iIBFunc(Pxy , Beta , Tsize , pi1 , pi2 , epsilon , LogKind)
% Calculate Information-bottleneck for a single variable case. Given the markov
% chain is Y-X-T, find the optimal T which minimize I(X,T) while maximize
% I(Y,T), the trade-off between the the two is determined by the Beta.
% Input: 
%   joint Dist. P(x,y)
%   Trade-off parameter Beta
%   Cardinality parameter Tsize
%   pi1 & pi2 - parameters for the Jensen divergence.
%   convergence parameter epsilon
% Output:
%   A soft partition P(t|x)
assert(ismembertol(sum(sum(Pxy)),1));
assert(Beta >= 0);
assert(Tsize > 0);
assert((pi1 + pi2) == 1);
Xsize = size(Pxy,1);
Ysize = size(Pxy,2);
Py_tr = sum(Pxy, 1);
Px_y = Pxy ./ repmat(Py_tr, Xsize, 1);
Px = sum(Pxy, 2);
Py_x = (Pxy ./ repmat(Px, 1, Ysize))';
assert(ismembertol(sum(Px),1));
%%
% Initialization - randomly initalize pt_x and find the corresponding Pt,
% Py_t
Pt_x = rand(Tsize, Xsize);
Pt_x = Pt_x ./ repmat(sum(Pt_x, 1), Tsize, 1) ; % Normalize the probability. division by zero is not possible since we divide by sum over T which must be larger than zero (T must get some value)
assert( all(ismembertol(sum(Pt_x,1), ones(1, Xsize) )) );
Ptx = Pt_x .* repmat(Px', Tsize, 1); 
Pt = sum(Ptx, 2);
assert(ismembertol(sum(Pt), 1));

Pt_y = Pt_x * Px_y;
Py_t = Pt_y' .* repmat(Py_tr', 1, Tsize) ./ repmat(Pt', Ysize, 1);
assert( all(ismembertol(sum(Py_t,1), ones(1, Tsize) )) );

%%
% Iterate untill the JS divergence(P^i+1(t|x), P^i(t_x)) < epsilon
max_ier = 1e5;
convergence_flag = false;
iter = 0;
while(~convergence_flag && (iter < max_ier))
   % Numeric errors are possible since we get very small numbers.
%    if(~all(all(-Beta * KLDivMat(Py_x, Py_t)>-200)))
%       assert(0); 
%    end
   Pt_x_new_unnorm = repmat(Pt,1, Xsize) .* exp(-Beta * KLDivMat(Py_x, Py_t));
   %Because division by zero is not possible since we divide by sum over T
   %which must be larger than zero (T must get some value). This can still happen
   %because rounding extermly small number. To protect agianst it I find
   %all colums which are equal to zero and replace all probabilities with
   %uniform probability
   Pt_x_new_unnorm(:,sum(Pt_x_new_unnorm) == 0)=1;
   
   Pt_x_new = Pt_x_new_unnorm ./ repmat(sum(Pt_x_new_unnorm, 1), Tsize, 1);
%    assert( all(ismembertol(sum(Pt_x_new,1), ones(1, Tsize) )) );
    if(~all(ismembertol(sum(Pt_x_new,1), ones(1, Tsize) )))
        keyboard;
    end
   
   Pt_new = Pt_x_new * Px;
   assert(ismembertol(sum(Pt_new), 1)); 
   
   Py_t_new = (Pt_x_new * Pxy ./ repmat(Pt_new + eps(Pt_new), 1, Ysize ))';
   assert(all(size(Py_t_new) == [Ysize Tsize]));
   assert(all(ismembertol(sum(Py_t_new,1), ones(1, Tsize))));
   
   if all(JSDiv(Pt_x_new, Pt_x, pi1, pi2) < epsilon)
    convergence_flag = true;
   end
   Pt_x = Pt_x_new;
   Pt = Pt_new;
   Py_t = Py_t_new;
   iter = iter + 1;
end
% fprintf('iIBFunc finished after %d iterations\n',iter);
%%
% Find the mutual information Ixt, Iyt
Ptx = Pt_x .* repmat(Px', Tsize, 1); 
Hxt  = sum(sum(-Ptx .* log2(Ptx)));
Ixt_mat = Ptx .* log2(Pt_x./repmat(Pt,1,Xsize));
Ixt_mat(isnan(Ixt_mat)) = 0;
Ixt = sum(sum(Ixt_mat));
Pyt = Py_t .* repmat(Pt', Ysize, 1);
Iyt = sum( sum(Pyt .* log2(Py_t./repmat(Py_tr', 1, Tsize) )));


end


function js_dist = JSDiv(P, Q, pi1 , pi2)
% js_dist = JSDiv(P, Q, pi1 , pi2)
% Calculate the Jensen-Shanon divergence
% Input:
%   P - Prob. matrix with dim TxX
%   Q - Prob. matrix with dim TxX
%   pi1 - the weight for the P distrbution
%   pi2 - the weight for the Q distrbution
    assert(isequal(size(P), size(Q)));
    
    M = pi1 * P + pi2 * Q;
    js_dist_mat = P.*log2(P./M) + Q.*log2(Q./M);
    js_dist = sum(js_dist_mat, 1);
end

function mat_dist = KLDivMat(P, Q)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  Input:
%   P - prob. matrix MxX - P(y|x)
%   Q - Prob. matrix MxT - P(y|t)
% Output:
%   mat_dist - a matrix TxN with each element containing the KL divergence
%   between 2 different colums permutations.

    assert(size(P,1) == size(Q,1));
    Tsize = size(Q,2);
    Xsize = size(P,2);
    assert(all(ismembertol(sum(P),ones(1,Xsize))));
    assert( all( ismembertol(sum(Q),ones(1,Tsize)) ) );

    
    H_P = sum(P .* log2(P)); % This is the entropy of P for a given x (with a minus sign missing)
    assert(all(size(H_P) == [1 Xsize])); 
    PlogQ = log2(Q') * P; %TxM * MxX = TxX
    mat_dist = repmat(H_P, Tsize,1) - PlogQ;  
    assert(all(size(mat_dist) == [Tsize Xsize]));
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