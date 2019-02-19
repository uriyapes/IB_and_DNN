function [Pyx, Px_y_tr, Py_new, Quantizer] = biKbarrier2Dmc(x_size, K, Py, shuffle_flag)
% Creates a K-barrier model P(x|y) for a DMC with inputs Y and outputs X.  
%   Example of 1-barrier model:
%       P(y=1|x) = P1 for x < barrier -> P(y=0|x) = 1 - P1 for x < barrier
%       P(y=1|x) = P2 for x > barrier -> P(y=0|x) = 1 -P2
%   Example of 2-barrier model:
%       P(y=1|x) = P1 for x < barrier1 -> P(y=0|x) = 1 - P1 for x <barrier1
%       P(y=1|x) = P2 for barrier1< x < barrier2 -> P(y=0|x) = 1 - P2 for same x range
%       P(y=1|x) = P3 for x > barrier2 -> P(y=0|x) = 1 - P3 for x >barrier3
% After drawing P(y|x) the simulation calculates P(x|y) = P(y|x)*P(x) / P(y),
% 
% Input:
%   x_size - the size of the X alphabet
%   K - the number of barriers in the model
% Output:
%   P is a 2-by-x_size matrix, where:
%      Px_y_tr = P(j,m) = Pr( X=m | Y=j )
  


if nargin < 1
    x_size = 20;
end
if nargin < 2
    K = 1;
end
if nargin < 4
    shuffle_flag = true;
end

assert(K <= x_size-1);

xmin = -2;
xmax = 2;

Quantizer = linspace(xmin,xmax,x_size);

% the boundries between each value of x bins
% Boundary = (Quantizer(1:end-1) + Quantizer(2:end))/2;


% dividers = xmin + (xmax-xmin)*rand(1,K);
% dividers = 0;
ind_without_tails = 1:size(Quantizer,2);
ind_without_tails = ind_without_tails(randperm(size(ind_without_tails,2), K));
quant_step = (xmax-xmin)/ x_size;
dividers = sort(Quantizer(ind_without_tails)) + 0.5*quant_step;
% Boundary = [-Inf Boundary Inf];
dividers = [-Inf dividers Inf];
interval_prob = rand(1,K+1);

Py_x = KbarrierProb(Quantizer, dividers, interval_prob);
if exist('Py', 'var') == 1
    % Py_x * Px = sum(Pyx,2) = Py 
    % Px = lsqnonneg(Py_x, Py);
    Aeq = ones(1,x_size);
    beq = 1;
    lb = 1/(10*x_size)*ones(1, x_size);
    ub = ones(1, x_size);
    options = optimoptions('lsqlin','Algorithm','interior-point','MaxIterations',1e6,'ConstraintTolerance', 1e-16,'OptimalityTolerance', 1e-16, 'StepTolerance',1e-18);
    Px = lsqlin(Py_x,Py,[],[],Aeq,beq,lb,ub,[],options)';
else
    Px = ones(1, x_size) / x_size;
end
if shuffle_flag
    rand_ind = randperm(size(Py_x,2));
else
    rand_ind = 1:size(Py_x,2);
end
Py_x_shuffled = Py_x(:,rand_ind);
Px = Px(rand_ind);
Pyx = Py_x_shuffled .* repmat(Px, 2, 1);
Py_new = sum(Pyx, 2);
Px_y_tr = Py_x_shuffled .* repmat(Px, 2, 1) ./ repmat(Py, 1, x_size);
assert(all(Px > 0));
assert(ismembertol(sum(Px,2), 1));
assert(all(ismembertol(sum(Px_y_tr,2), ones(2,1))));
assert(all(ismembertol(Py_new, Py)));
assert(all(ismembertol(sum(sum(Pyx)), 1)));
Pyx = Px_y_tr .* repmat(Py, 1, x_size);

% Boundary = Quantizer;
%%
figure(99)
subplot(4,1,1);
stem(Quantizer, Py_x(1,:))
title('P(y=1|x) unshuffled');
subplot(4,1,2);
stem(Quantizer, Py_x_shuffled(1,:))
title('P(y=1|x)');
subplot(4,1,3);
stem(Quantizer, Px)
title('P(x)')
% subplot(4,1,3);
% stem(Py_new)
% title('P(y)')
subplot(4,1,4);
stem(Quantizer, Pyx(1,:))
title_txt = sprintf('P(x,y=1)   P(y=1)=%d',Py(1));
title(title_txt)

% % The following method generate random x, with it Pyx is created, then
% % using the constraint over Py=[0.5 0.5] a new Pyx is created by normalaizing old Pyx, the new Pyxhave
% % correct Py marginals. Because Pyx was created by Py_x and then divided by
% % a 2X1 vector it didn't change the barrier structure of Py_x and therefore
% % the new Py_x keeps it shape.
% Px_unnorm = rand(1, x_size);
% Px = Px_unnorm ./ sum(Px_unnorm,2);
% Pyx_unnorm = Py_x .* repmat(Px, 2, 1); 
% % Pyx_unnorm result will not have marginal probability of Py = [0.5 0.5],
% % lets take care of it.
% 
% Pyx = Pyx_unnorm ./ (2*sum(Pyx_unnorm,2));
% assert(all(ismembertol(sum(Pyx,2), Py)));
% Px = sum(Pyx,1);
% assert(ismembertol(sum(Px,2), 1));
% Py_x_new =  Pyx ./ repmat(Px, 2, 1);
% Px_y_tr = Py_x_new .* repmat(Px, 2, 1) ./ repmat(Py, 1, x_size);
% assert(all(ismembertol(sum(Px_y_tr,2), ones(2, 1))));




% K barrier function
function P = KbarrierProb(Boundary, dividers, interval_prob)
% KbarrierProb creates P(y|x) according to the K-barrier model.
% Input:
%   Boundary - vector of the X values
%   dividers - vector representing the location of each one of the barriers
%   interval_prob - a  probability vector which determines the probability in each interval created
%                   by the barriers.
% Output:
%   P - a matrix of P(y|x)

    K = length(dividers);
    P = zeros(2, length(Boundary));
    
    %Calculate P(y|x)
    for j = 1:K-1
        ind = (Boundary>=dividers(j) & Boundary<dividers(j+1));
        if((j==K-1) && sum(ind,2)==0)
            continue
        end
        assert(sum(ind,2) ~= 0);
        assert(sum(sum(any(P(:,ind)))) == 0);
        P(1,ind) = interval_prob(j);
        P(2,ind) = 1 - P(1,ind);
    end
    assert(all(ismembertol(sum(P,1), ones(1, size(Boundary,2)))));


