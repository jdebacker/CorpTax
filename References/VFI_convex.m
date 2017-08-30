%%% This program solves the firm's Bellman equation, solved market
%%% clearing, and finds the stationary distribution of firms, output is the
%%% moments from the guess at the structural parms

function [modelM] = VFI_convex(parameters) 

global L Q cost_type exogenous_w exog_wage guesses_parms guesses guesses_moments guesses_eqm guesses_minstat counter 
global dataM spike_level equity_spike_threshold tau_i tau_d tau_g tau_c beta delta L_bar lb_s r 
global betafirm sizez mu_z Pi zvec sizek ub_k lb_k kvec dens VFflag 
global MCleared MC_wage VF Weight nonconvex_cost

if L < 8 
   global rho sigma_z alpha_k alpha_l
end

if cost_type == 1
    global psi lambda FC phi0 phi1 
elseif cost_type == 2 
    global lambda FC phi0 phi1 
    psi = parameters(1,1) ;
elseif cost_type == 3 
    global lambda FC phi1 
    psi  = parameters(1,1) ; 
    phi0 = parameters(2,1) ;
elseif cost_type == 4 
    global lambda FC
    psi  = parameters(1,1) ; 
    phi0 = parameters(2,1) ;
    phi1 = parameters(3,1) ;
elseif cost_type == 5 
    global phi0 phi1 
    psi = parameters(1,1) ;
    if nonconvex_cost == 0
        global FC
        lambda  = parameters(2,1) ;
    else
        global lambda
        FC = parameters(2,1) ;
    end
elseif cost_type == 6 
    global phi1
    psi  = parameters(1,1) ;
    if nonconvex_cost == 0
        global FC
        lambda  = parameters(2,1) ;
    else
        global lambda
        FC = parameters(2,1) ;
    end
    phi0 = parameters(3,1) ;
elseif cost_type == 7 
    psi  = parameters(1,1) ;
    if nonconvex_cost == 0
        global FC
        lambda  = parameters(2,1) ;
    else
        global lambda
        FC = parameters(2,1) ;
    end
    phi0 = parameters(3,1) ;
    phi1 = parameters(4,1) ;
elseif cost_type == 9 
    psi  = parameters(1,1) ;
    if nonconvex_cost == 0
        global FC
        lambda  = parameters(2,1) ;
    else
        global lambda
        FC = parameters(2,1) ;
    end
    phi0 = parameters(3,1) ; 
    phi1 = parameters(4,1) ;
    rho = parameters(5,1) ;
    sigma_z = parameters(6,1) ;
    alpha_l = parameters(7,1) ;
    alpha_k = parameters(8,1) ;
end



%-------------------------------------------------------------------------%
% Discretizing state space for productivity shocks                             %
%-------------------------------------------------------------------------%
% sizez   = fineness of grid for firm productivity shocks
% mu      = unconditional mean of productivity shocks
% rho     = persistence of productivity
% sigma_z = std dev of shocks to AR(1) process for firm productivity
%-------------------------------------------------------------------------%
if L == 8
% % tauchen.m will use the Tauchen (1991) method to approximate a
% % continuous distribution of shocks to the AR1 process with a Markov process.
% [Pi,eps,z]=tauchen(sizez,mu,rho,sigma_z) ;
% %z = exp(z) ;
% zvec = z ; % grid of productivity values

% %Try with Sargent's version of Tauchen %
% % tauchen from Sargent %
% m = 3.1 ; % number of std dev is width 
% [Pi,z,probst,alambda,asigmay]=markovappr(rho,sigma_z,m,sizez) ;
% Pi = Pi' ;
% z = exp(z) ;
% zvec = z' ;

% % Tauchen with Gomes' program
    [Pi, z] = Quadnorm(sizez, mu_z, sigma_z, rho);
    Pi = Pi' ;
    zvec = z' ;
end


%-------------------------------------------------------------------------%
% General Equilibrium Loop                                                %
%-------------------------------------------------------------------------%
% w             = real wage
% MCtol         = tolerance for market clearing condition
% MCmaxiter     = maximum alpha_lmber of iterations allowed to find market clearing
%   wage
% currentMCdist = MC distance for current iteration
% MCflag        = flag = 0 until go "past" correct wage and need to switch
%   direction of search for market clearing wage
%-------------------------------------------------------------------------%

if exogenous_w == 0
    if MCleared == 0
        w = 1.3 ; %0.001 ; % Initial guess at wage  
    else
        w = MC_wage ; % make first guess the wage that cleared with the last guess of parameters
    end
    MCmaxiter = 200 ; % maximum iterations to find market clearing wage
else
    w = exog_wage ; % exogenous wage
    MCmaxiter = 2 ; % maximum iterations to find market clearing wage- make it so quit with exogenous wage 
end
MCtol = 10^(-5) ;
MCiter = 1 ;
currentMCdist = 7 ;
MCflag = 0 ; % flag =0 until go past correct price
while currentMCdist>MCtol & MCmaxiter>MCiter


%-------------------------------------------------------------------------%
% Generating possible cash flow levels                                    %
%-------------------------------------------------------------------------%
% op = operating profits, a two dimensional array of profits for each
% combination of capital stock and productivity shock
% C  = 3-dimensional array of cash flows, a cash flow value is calculated
% for each possible combination of capital stock today (state) and
% productivity shock (state), and choice of capital stock tomorrow
% (control) - note that dividends and share issues/repurchases are
% determined from the BC and the optimal choice of capital
%-------------------------------------------------------------------------%

op = (1-alpha_l)*((alpha_l/w)^(alpha_l/(1-alpha_l)))*((kron((kvec'.^alpha_k),exp(zvec))).^(1/(1-alpha_l))) ;

%%% make 3 dimensional so can add and such
op3 = repmat(op,[1,1,sizek]) ;
k3 = repmat(kvec',[1,sizez,sizek]) ;
kp3 = permute(repmat(kvec',[1,sizez,sizek]),[3 2 1]) ;


C = ((1-tau_c)*op3) + (delta*tau_c*k3) - kp3 + ((1-delta)*k3) - ((psi/2)*((kp3-((1-delta)*k3)).^2)./k3) ;




    %-------------------------------------------------------------------------%
    % Value Function Iteration                                                %
    %-------------------------------------------------------------------------%
    % VFtol     = tolerance required for value function to converge
    % VFdist    = distance between last two value functions
    % VFmaxiter = maximum alpha_lmber of iterations for value function
    % VFiter    = current iteration alpha_lmber
    % EVmat     = array whose last two dims are the expected value of the value function 
    %             values from the last iteration
    % VF        = matrix of maximized value functions for each level of
    %             past ideology and voter ideology
    % PF        = matrix of indicies of Senator choices for all states
    %             (i.e. all past ideology and voter ideology)
    %-------------------------------------------------------------------------%
            VFtol = 10^(-6) ;
			VFdist = 7 ;
		    if VFflag == 1
                V = VF ; % make initial guess the VF from the last MC iteration/last guess at parms
            else
                V = zeros(sizek,sizez) ; % initial guess at value function
            end
			VFmaxiter = 2000 ; 
			VFiter = 1 ;
            
            while VFdist > VFtol & VFiter<VFmaxiter
				TV = V  ;
                %Create matrix for expected value of position tomorrow
                EV = (Pi*TV')' ;
                EVmat = permute(repmat(EV,[1,1,sizek]),[3,2,1]) ;
                Vmat = ((1-tau_d)/(1-tau_g)).*(C.*(C>=0)) + C.*(C<0)+((C<0).*phi1.*C) - (phi0*(C<0)) + (betafirm*EVmat) ;
                [V, PF_continuous] = max(Vmat,[],3) ;
                VFdist = max(max(max(abs(V-TV)))) ; 
                VFiter = VFiter+1 ;
            end


%  if VFiter<VFmaxiter
%                disp('Value function converged after this many iterations:')
%               disp(VFiter)
%  else
%                disp('Value function did not converge')              
%  end

  VF = V ;
     
  VFflag = 1 ;
  
    
  PF_discrete = zeros(sizek,sizez) ; % because not nonconvexitites- important in tax_moments.m
 
  

    %-------------------------------------------------------------------------%
    % Find Stationary Distribution                                            %
    %-------------------------------------------------------------------------%
    % SDtol     = tolerance required for convergence of SD
    % SDdist    = distance between last two SDs
    % SDiter    = current iteration
    % SDmaxiter = maximium iteration allowed to find stationary distribution
    % mu        = stationary distribution 
    % Tmu       = operated on stationary distribution
    %-------------------------------------------------------------------------%
    mu    = ones(sizek,sizez).*(1/(sizek*sizez));
    SDtol  = 10^(-12) ;
    SDdist = 1 ;
    SDiter = 0 ;
    SDmaxiter = 1000 ;
    while SDdist > SDtol & SDmaxiter > SDiter 
        Tmu = zeros(sizek,sizez);
        for i=1:sizek %capital stock
            for j=1:sizez %productivity
                for jj=1:sizez % productivity next period
                    Tmu(PF_continuous(i,j),jj)=Tmu(PF_continuous(i,j),jj)+Pi(j,jj)*mu(i,j);
                end
            end
        end
        SDdist = max(max(abs(Tmu-mu)));
        mu    = Tmu;
        SDiter = SDiter+1 ;
    end 

%  if SDiter<SDmaxiter
%                disp('Stationary distribution converged after this many iterations:')
%               disp(SDiter)
%  else
%                disp('Stationary distribution did not converge')              
%  end


 
    %-------------------------------------------------------------------------%
    % Market Clearning                                                        %
    %-------------------------------------------------------------------------%
    % labor = labor demand for each combination of kaptial and productivity
    % l     = aggregate labor demand from the stationary distribution of
    %  firms
    % w_change = increment to change labor supply by until go past labor
    %   clearing quantity
    % w1    = upper bound from the previous guesses
    % w0    = lower bound from the previous guesses
    %-------------------------------------------------------------------------%
		
		
    agg_labor(MCiter) = sum(sum(((((alpha_l/w)^(1/(1-alpha_l)))*((kron((kvec'.^alpha_k),exp(zvec))).^(1/(1-alpha_l)))).*mu)))  ; 	

        
	wguess(MCiter) = w ;	       
        %%%%%%%%%%% Check to see if markets clear %%%%%%%%%%%%%
        %%% simple bisection since this 'excess demand' appears to be
        %%% always monotonic


 MCdist(MCiter) = agg_labor(MCiter) - L_bar ;
        wguess(MCiter) = w ;
        w_change = 0.5 ;
        if MCiter == 1
           w0 = w ;
           if MCdist(MCiter) > 0 
               w = w0 + w_change ;
           elseif MCdist(MCiter) < 0 
               w = max(0.00001, (w0 - w_change)) ;
           end
        elseif MCiter >= 2
            if MCflag == 0 
                if MCdist(MCiter) < 0 & MCdist(MCiter-1) > 0
                    MCflag = 1 ;
                    wup = w ;
                    wdown = w0 ;
                    w = (wup+wdown)/2 ;
                elseif MCdist(MCiter) > 0 & MCdist(MCiter-1) < 0
                    MCflag = 1 ;
                    wdown = w ;
                    wup = w0 ;
                    w = (wup+wdown)/2 ;
                elseif MCdist(MCiter) > 0 & MCdist(MCiter) > 0
                    w0 = w ;
                    w = w0 + w_change ;
                elseif MCdist(MCiter) < 0 & MCdist(MCiter) < 0 
                    w0 = w ;
                    w = max(0.00001, (w0 - w_change)) ;
                end
            elseif MCflag == 1
                if MCdist(MCiter) > 0
                    wup = wup ;
                    wdown = w ;
                    w = (wup+wdown)/2 ;
                elseif MCdist(MCiter) <0 
                    wdown = wdown ;
                    wup = w ;
                    w = (wup+wdown)/2 ;
                end
            end
        end
        
        
	currentMCdist = abs(MCdist(MCiter)); % want a scalar for the current distance comparison
%     disp('MC distance')
%     MCdist(MCiter)
%     disp('MC iteration')
    MCiter = MCiter +1 ;
%     disp('Wage this iteration')
%     w 
end %End market clearing loop
 
if exogenous_w == 1
    w = exog_wage ;
end

 if MCiter < MCmaxiter
    %disp('Convergence of market clearing achieved after this many iterations:')
    %disp(MCiter)
    MC_wage = w ;
    MCleared = 1 ;
else
    %disp('Convergence of market clearing not achieved')
 end
 
if MCiter >= MCmaxiter
    guesses_eqm(3,counter) = 1 ;
end
if SDiter >= SDmaxiter
    guesses_eqm(2,counter) = 1 ;
end
if VFiter >= VFmaxiter
    guesses_eqm(1,counter) = 1 ;
end
if sum(mu(:,sizez)) >= 0.999
    guess_eqm(4,counter) = 1 ;
end

modelM = tax_moments(parameters,PF_continuous,PF_discrete,V,C,w,mu,op) ;


    
    
    
    
    
 
    
    

     