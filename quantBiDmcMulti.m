function [Qout,mi,Pxy] = quantBiDmcMulti(Pxy, K)
% [Q,MI] = quantBiDmcMulti(P,K)
%
% Finds the optimal quantizer of DMC given by P, quantized to K values,
% in the sense of maximizing mutual information.
% P is a 2-by-M matrix, where:
%   P(j,m) = Pr( Y=m | X=j ),
% for a DMC with inputs X and outputs Y.  K is an integer, generally less
% than M. 
%
% Q is a cell array containing M-by-K matrices (MEK: it seems like K by M matrix)
%  each one is an optimal quantizer.
% For each matrix Q{i}, Q{i}(m,k) is a 1 if DMC output m is quantized
% to k, and otherwise is a 0.
%
% MI is the mutual information between the channel input and the quantizer
% output, which is the same for all optimal quantizers.
%
% QuantDMC (c) Brian Kurkoski and contributors
% Distributed under an MIT-like license; see the file LICENSE
%

J = size(Pxy,1);
I = size(Pxy,2);
if J ~= 2
    error('This only works with binary matrices');
end

if K >= I
	assert(0); %It should be T=P(Y|X) and not Pxy
    T = Pxy;
    mi = NaN;
    Qout = {eye(size(Pxy,2))};
    return
end

%Input Pxy is conditional, but this function assumes Pxy is joint.
% Pxy = Pxy / 2;

%Pxy is joint, construct Pcond to sort.
Pcond = Pxy ./ repmat( sum(Pxy,2),1,I);
LLR = log( Pcond(1,:) ./ Pcond(2,:) );
[t,sortorder] = sort(LLR);

%Sorted joint probability P is used from now on
P = Pxy(:,sortorder);
sorted_Pyx = P;


%initial distance computation (step 3 - Eq. 24 or 25)
dist = zeros(I,I);
pj = sum(P,2);%gives Px
for ii = 1:I %choose a'
    for kk = ii:I %choose a 
        t = sum(P(:,ii:kk),2);%sigma over p(x,y') in a range of y' between a'+1 to a
        s = sum(t); % sigmaX(Px'*SigmaY_a'+1_a(P(y|x')) = Sigma over all x'(sigma over y' between a'+1 to a(P(x',y'))
        %if s > 1; s =1; end
        
        % dvision by pj (which is P(x)) is done so the numerator inside the log will be sigma over P(y|x) and not over P(x,y). 
        % The sum over the t variable gives in fact sigma(t) =
        % sigmaX(sigmaY_a'+1_a(P(x,y)) = sigmaX(Px * sigmaY_a'+1_a(P(y|x))
        dist(ii,kk) = sum( (t .* log2(t ./ (s * pj) ) ) +eps );
    end
end

SM = zeros(I,K);
ps = cell(I,K);
SM(:,1) = dist(1,:);
fl = 0;
for kk = 2:K %for each z between [1...K], why 2?
    for ii = kk:I %for each a between [1...M]
        t = zeros(size([kk-1:ii-1]) );
        for ell = kk-1:ii-1 %ell is a', so what is done here is to calc. all hz(a) with regard to a'
            t(ell - (kk-2) ) =  SM(ell,kk-1) + dist(ell+1,ii) ;
        end
        [SM(ii,kk),ps{ii,kk}] =  max(t);
        %ps(ii,kk) = ps(ii,kk) + kk - 2;
        ps{ii,kk} =  find(t == max(t));
        ps{ii,kk} = ps{ii,kk} + kk - 2;
        %if length(ps{ii,kk}) > 1
        %    fl = 1;
        %end
    end
end

%build quantizer list
Q = [I];
for kk = K:-1:1
    Qnew = [];
    for ii = 1:size(Q,1)
        s = Q(ii,K-kk+1);
        t = ps{s,kk};
        if length(t) == 0;
            t = 0;
        end
        if length(t) == 1
            Q(ii,K-kk+2) = t;
        else
            Q(ii,K-kk+2) = t(1);
            Qt = repmat(Q(ii,1:K-kk+1),length(t)-1,1);
            tp = t(:);
            Qt = [Qt tp(2:end)];
            Qnew = [Qnew; Qt];
        end
    end
    Q = [Q; Qnew];
end

%build the quantizer from the quantizer list
Qlist = Q;
Q={};
for ii = 1:size(Qlist,1)
    Q{ii} = zeros(K,I);
    for kk = 1:K
        Q{ii}(K-kk+1, Qlist(ii,kk+1)+1:Qlist(ii,kk)) = 1;
    end
end

%compute mutual information associated with each quantier.
for ii = 1:length(Q)
    %See Eq.2 but here T is the joint prob. of x and z - Pzx=sigmaY(P(z,x|y)Py)=
    %sigmaY(P(z|x,y)P(x|y)Py) = sigmaY(P(z|y)Pxy))
    T = P * Q{ii}' ; 
    p=sum(T,2);
    q=sum(T,1);
    mi(ii)=0;
    for kk = 1:K
        for jj = 1:J
            % Simillar to Eq. 3. but with T as the joint Prob., this is why
            % we dont need to multiply by px before the T.
            mi(ii) = mi(ii) + T(jj,kk) * log2( T(jj,kk) / (q(kk) * p(jj))) ;
        end
    end
end
%disp([ SM(I,K) mi])

%reverse the sort order to agree with Pxy input
for ii = 1:length(Q)
    Q{ii}(:,sortorder) = Q{ii};
end    

Qout = Q;
Pzx = (Pxy * Q{1}')'; 

return 
