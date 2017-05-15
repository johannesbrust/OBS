function [sigmaStar,pStar,opt1,opt2,spd_check] = obs( g, S, Y, delta, gamma,varargin);

% Computes the solution to a trust-region subproblem when the quadratic
% model is defined by limited-memory symmetric rank-one (L-SR1)
% quasi-Newton matrix.
%
% "OBS: MATLAB Solver for L-SR1 Trust-Region Subproblems"
% by Johannes Burst, Jennifer Erway, and Roummel Marcia
%
% Copyright (2015): Johannes Brust, Jennifer Erway, and Roummel Marcia
%
% The technical report and software are available at 
% www.wfu.edu/~erwayjb/publications.html
% www.wfu.edu/~erwayjb/software.html
%
% This code is distributed under the terms of the GNU General Public
% License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire 
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%---------------------------------------------------------------------------    
% 
% Inputs: g: the gradient vector and right-hand side of the first
%                optimality condition
%         S,Y: the quasi-Newton pairs that define the SR1 matrix
%         delta: the trust-region radius
%         gamma: scaling of initial Hessian matrix approximation B0, 
%                i.e., B0 = gamma*I
%         varargin: In lieu of S and Y, Psi and invM, in this order may be
%                   given.  In this case, S and Y can be set to [ ].
%
% Output: sigmaStar: the optimal Lagrange multiplier
%         pStar: the optimal solution to the trust-region subproblem
%         opt1: the residual of the 1st optimality condition: norm(BpStar+sigmaStar*pStar + g) 
%         opt2: the residual of the 2nd optimality condition: sigmaStar*abs(delta-norm(pStar))
%         spd_check: gives the smallest eig of (B+sigmaStar I): lambda_min + sigmaStar 
%
% Note: the variable "show" can be toggled as follows:
%       show = 0 runs silently and assigns no value to opt1, opt2, spd_check
%            = 1 runs silently but assigns value to opt1, opt2, spd_check
%            = 2 verbose setting; also assigns value to opt1, opt2, spd_check
%
show        = 1;     % verbosity flag


if ~((nargin==5) | (nargin==7))
  ferror('\nWrong number of inputs...');
elseif (nargin==7)
  Psi        = varargin{1};
  invM       = varargin{2};
end

%% Initializations
maxIter     = 100;   % maximum number of iterations for Newton's method
tol         = 1e-10; % tolerance for Newton's method

if (nargin==5)  %inputs were S and Y
  %% Compute S'Y, S'S, inv(M), Psi
  SY          = S'*Y;
  SS          = S'*S;
  invM        = tril(SY)+ tril(SY,-1)'-gamma.*SS;
  invM        = (invM+(invM)')./2;  %symmetrize invM, if needed
  Psi         = Y - gamma.*S;
end

%% Ensure Psi is full rank
if (rank(Psi)~=size(Psi,2))
  fprintf('\n\n');
  ME = MException('Catastropic_error:Psi',' Psi is not full rank... exiting obs.m');
  throw(ME);
end

%% Compute eigenvalues Hat{Lambda} using Choleksy
PsiPsi      = Psi'*Psi;
R           = chol(PsiPsi);
RMR         = R* (invM\R'); 
RMR         = (RMR+(RMR)')./2;  %symmetrize RMR', if needed
[U D ]      = eig(RMR);

%% Eliminate complex roundoff error then sort eigenvalues and eigenvectors
[D,ind]     = sort(real(diag(D)));  %D is now a vector
U           = U(:,ind);            

% Compute Lambda as in Equation (7) and lambda_min
sizeD       = size(D,1);
Lambda_one  = D + gamma.*ones(sizeD,1);
Lambda      = [ Lambda_one; gamma]; 
Lambda      = Lambda.*(abs(Lambda)>tol); %thresholds
lambda_min  = min([Lambda(1), gamma]); 

% Define P_parallel and g_parallel 
RU          = R\U;     
P_parallel  = Psi*RU;  
Psig        = Psi'*g;  
g_parallel  = RU'*Psig; 

% Compute a_j = (g_parallel)_j for j=1...k+1; a_{k+2}=||g_perp||
a_kp2      = sqrt(abs(g'*g - g_parallel'*g_parallel));
if a_kp2^2 < tol  %fix small values of a_kp2 
  a_kp2 = 0;
end
a_j        = [ g_parallel; a_kp2 ];

% (1) Check unconstrained minimizer p_u
if (lambda_min>0) & (norm(a_j./Lambda)<=delta)  
  sigmaStar = 0;
  pStar     = ComputeSBySMW(gamma,g,Psig,Psi,invM,PsiPsi);
elseif (lambda_min<=0) & (phiBar_f(-lambda_min,delta,Lambda,a_j)>0)
  sigmaStar = -lambda_min;

  % forms v = (Lambda_one + sigmaStar I)^\dagger P_\parallel^Tg
  index_pseudo = find(abs(Lambda+sigmaStar)>tol);
  v = zeros(sizeD+1,1);
  v(index_pseudo) = a_j(index_pseudo)./(Lambda(index_pseudo)+sigmaStar); 
  
  % forms pStar using Equation (16)
  if abs(gamma+sigmaStar)<tol
      pStar = -P_parallel*v(1:sizeD); 
  else
    pStar = -P_parallel*v(1:sizeD) + (1/(gamma+sigmaStar)).*(Psi*(PsiPsi\Psig)) - (g./(gamma+sigmaStar));
  end

  if lambda_min < 0
    alpha = sqrt(delta^2-pStar'*pStar);
    pHatStar = pStar;
    
    % compute z* using Equation (17)
    if abs(lambda_min-Lambda(1))<tol 
      zstar = (1/norm(P_parallel(:,1))).*alpha*P_parallel(:,1); 
    else %gamma=lambda_min
      e = zeros(size(g,1),1);
      found = 0; 
      for i=1:sizeD
        e(i)=1;
        u_min = e- P_parallel*P_parallel(i,1:end)';
        if norm(u_min)>tol
            found = 1;
            break;
        end
        e(i)=0;
      end  
      if found == 0
        e(m+1) = 1;
        u_min = e- P_parallel*P_parallel(i,1:end);
      end
      u_min = u_min/(norm(u_min));
      zstar = alpha*u_min;
    end
    pStar = pHatStar+zstar; 
  end
  
else
  if lambda_min>0
    sigmaStar = Newton(0,maxIter,tol,delta,Lambda,a_j);
  else
    sigmaHat = max(a_j/delta - Lambda);
    if sigmaHat>-lambda_min 
      sigmaStar = Newton(sigmaHat,maxIter,tol,delta,Lambda,a_j);
    else
      sigmaStar = Newton(-lambda_min,maxIter,tol,delta,Lambda,a_j);
    end
  end
  pStar     = ComputeSBySMW(gamma+sigmaStar,g,Psig,Psi,invM,PsiPsi);
end

%%% optimality check
if show>=1
  BpStar = gamma.*pStar + Psi*(invM\(Psi'*pStar));  %uses Equation (4)
  opt1   = norm(BpStar+sigmaStar*pStar + g);
  opt2   = sigmaStar*abs(delta-norm(pStar));
  spd_check = lambda_min + sigmaStar;
  if show==2
    fprintf('\nOptimality condition #1: %8.3e', opt1);
    fprintf('\nOptimality condition #2: %8.3e', opt2);
    fprintf('\nlambda_min+sigma*: %8.2e', spd_check);
    fprintf('\n\n');
  end
else
  opt1 = [ ];
  opt2 = [ ];
  spd_check = [ ];
  phiBar_check = [ ];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
function [ pStar ] = ComputeSBySMW( tauStar, g, Psig, Psi, invM, PsiPsi)
%ComputeSBySWM solves (B+tauStar)sStar = -g for sStar where B is
%an L-SR1 matrix using the Sherman-Morison-Woodbury formula.
%In particular,
% s* = -[ 1/tau*( I - (tau*.invM + PsiPsi)^(-1))Psi') ][ g ].
%
% For further details, please see the following technical report:
%
% "OBS: MATLAB Solver for L-SR1 Trust-Region Subproblems"
% by Johannes Burst, Jennifer Erway, and Roummel Marcia
%
% Copyright (2015): Johannes Brust, Jennifer Erway, and Roummel Marcia
%
% The technical report and software are available at 
% www.wfu.edu/~erwayjb/publications.html
% www.wfu.edu/~erwayjb/software.html
%
% This code is distributed under the terms of the GNU General Public
% License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire 
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%---------------------------------------------------------------------------    

vw              = tauStar^2.*invM + tauStar.*PsiPsi;
pStar            = -g./tauStar + Psi*(vw\Psig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ phiBar ] = phiBar_f( sigma,delta,D,a_j )
%phiBar_f evaluates the continous extension of 
%phi(sigma) = 1/ ||v|| - 1/delta. 
%
%For further details, please see the following technical report:
%
% "OBS: MATLAB Solver for L-SR1 Trust-Region Subproblems"
% by Johannes Burst, Jennifer Erway, and Roummel Marcia
%
% Copyright (2015): Johannes Brust, Jennifer Erway, and Roummel Marcia
%
% The technical report and software are available at 
% www.wfu.edu/~erwayjb/publications.html
% www.wfu.edu/~erwayjb/software.html
%
% This code is distributed under the terms of the GNU General Public
% License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire 
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%---------------------------------------------------------------------------    


m = size(a_j,1); 
D = D + sigma.*ones(m,1);   %vector
eps_tol = 1e-10;

% test if numerator or denominator has a zero
if ( sum( abs(a_j) < eps_tol ) > 0 ) | (sum( abs(diag(D)) < eps_tol ) > 0 )    
  pnorm2 = 0;
  for i = 1:m        
    if (abs(a_j(i)) > eps_tol) & (abs(D(i)) < eps_tol)
      phiBar = -1/delta;
      return; 
    elseif (abs(a_j(i)) > eps_tol) & (abs(D(i)) > eps_tol)
      pnorm2    = pnorm2 + (a_j(i)/D(i))^2;
    end
  end
  phiBar     = sqrt(1/pnorm2) - 1/delta;
  return;
end

%% numerators and denominators are nonzero
p = a_j./D;
phiBar = 1/norm(p) - 1/delta;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ phiBar, phiBar_g ] = phiBar_fg( sigma,delta,D,a_j )
%phiBar_f evaluates the continous extension of 
%phi(sigma) = 1/ ||v|| - 1/delta and its derivative.
%
%For further details, please see the following technical report:
%
% "OBS: MATLAB Solver for L-SR1 Trust-Region Subproblems"
% by Johannes Burst, Jennifer Erway, and Roummel Marcia
%
% Copyright (2015): Johannes Brust, Jennifer Erway, and Roummel Marcia
%
% The technical report and software are available at 
% www.wfu.edu/~erwayjb/publications.html
% www.wfu.edu/~erwayjb/software.html
%
% This code is distributed under the terms of the GNU General Public
% License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire 
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%---------------------------------------------------------------------------    

m        = size(a_j,1); 
D        = D + sigma.*ones(m,1);   %vector
eps_tol  = 1e-10; 
phiBar_g = 0;

% test if numerator or denominator has a zero
if ( sum( abs(a_j) < eps_tol ) > 0 ) || (sum( abs((D)) < eps_tol ) > 0 )    
  pnorm2 = 0;
  for i = 1:m        
    if (abs(a_j(i)) > eps_tol) && (abs(D(i)) < eps_tol)
      phiBar   = -1/delta; 
      phiBar_g = 1/eps_tol;
      return; 
    elseif abs(a_j(i)) > eps_tol && abs(D(i)) > eps_tol
      pnorm2   = pnorm2   +  (a_j(i)/D(i))^2;
      phiBar_g = phiBar_g + ((a_j(i))^2)/((D(i))^3);
    end
  end
  normP    = sqrt(pnorm2);
  phiBar   = 1/normP - 1/delta;
  phiBar_g = phiBar_g/(normP^3);
  return;
end

%%% Numerators and denominators are all nonzero
% Compute phiBar(sigma)
p      = a_j./D;
normP  = norm(p);
phiBar = 1/normP - 1/delta;

phiBar_g = sum((a_j.^2)./(D.^3));
phiBar_g = phiBar_g/(normP^3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ x ] = Newton(x0,maxIter,tol,delta,Lambda,a_j)
%Newton finds a zero of phiBar.
%
%For further details, please see the following technical report:
%
% "OBS: MATLAB Solver for L-SR1 Trust-Region Subproblems"
% by Johannes Burst, Jennifer Erway, and Roummel Marcia
%
% Copyright (2015): Johannes Brust, Jennifer Erway, and Roummel Marcia
%
% The technical report and software are available at 
% www.wfu.edu/~erwayjb/publications.html
% www.wfu.edu/~erwayjb/software.html
%
% This code is distributed under the terms of the GNU General Public
% License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire 
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%---------------------------------------------------------------------------    


x = x0;  %initialization
k = 0;   %counter

[f g]  = phiBar_fg(x,delta,Lambda,a_j);

while (abs(f) > eps) & (k < maxIter)
    x       = x - f / g;
    [f g]  = phiBar_fg(x,delta,Lambda,a_j);
    k = k + 1;
end



