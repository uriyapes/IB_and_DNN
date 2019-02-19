function [Pt_x_mat , Ixt_vec , Iyt_vec , Ptx_mat , lastUpdate, Py_t_mat, convergence_flag_vec] = iterativeiIB(PXy , beta , Tsize , pi1 , pi2 , epsilon , LogKind , iterationNumber)
% This function is iterating the "iIBFunc" from different starting points.  
% Since the initialization of the iIB algorithm is random,
% the algorithm starts at some random points in the
% mapping space (space of all mapping functions). The iIB algorithm
% will converge to a local minimum at this space. In order to be sure
% about obtaining the best answer the algorithm is iterated
% to start from different random points and for each result the 
% functional "IXt - Beta * Iyt" is calculated and the mapping
% responsible to the minimum functional value 
% among all the results is considered as the final result.
% The inputs 
%
% PXy, Tsize, pi1, pi2, epsilon, LogKind
% are correspond to the iIBFunc function. and the input
%
% "iterationNumber"         is the number of times we want to 
%                           iIB algorithm to run.
%
% "Beta"                    is a vector which correspond to the beta values
%                           we want to check along the IB curve
%
% OUTPUTS:
%
% "Pt_Xfinal"               is the best mapping obtained 
%                           from the all runs correspond to minimizing
%                           the functional IXt - Beta * Iyt
%
% "IXtfinal"                I(X;t) correspond to Pt_Xfinal
%
% "Iytfinal"                I(y;t) correspond to the final result.
%
% "lastUpdate"              the iteration number where the last update
%                           accured.


for b=1:size(beta, 2)
    Lfunctional = inf;
    ImprovementFlag = 0;
    iteration = 1;
    while iteration<=iterationNumber
        [Pt_X , IXt , Iyt , PtX, Py_t, convergence_flag] = iIBFunc(PXy , beta(b) , Tsize , pi1 , pi2 , epsilon , LogKind);

        if Lfunctional > (IXt - beta(b) * Iyt)
            Lfunctional = IXt - beta(b) * Iyt;
            Pt_Xfinal = Pt_X;
            PtXfinal = PtX;
            Py_t_final = Py_t;
            IXtfinal = IXt;
            Iytfinal = Iyt;
            lastUpdate = iteration;
            convergence_flag_final = convergence_flag;
            ImprovementFlag = 1;
        end
        %ImprovmentFlag should always be true
        if ImprovementFlag == 0
            assert(0);
        end
        %speed up calculation, if the iteration didn't converge then
        %increase iteration by 2 so there will be less iterations.
        if convergence_flag==1
            iteration = iteration + 1;
        else
            iteration = iteration + 2;
        end
    end
    Pt_x_mat(:,:,b) = Pt_Xfinal;
    Ptx_mat(:,:,b) = PtXfinal;
    Py_t_mat(:,:,b) = Py_t_final;
    Ixt_vec(b) = IXtfinal;
    Iyt_vec(b) = Iytfinal;
    convergence_flag_vec(b) = convergence_flag_final;
    fprintf('convergence_flag_final = %d\n', convergence_flag_final);
end
