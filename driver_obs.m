function [ ] = driver_obs( );
% driver for obs.m
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

clear all;
clc; 

%constants
n = logspace(3,7,5);  %generates problem sizes (1e3 to 1e7)
kp1 = 5;  %number of limited memory updates

n_runs = 5;  % number of test runs for testing time
i_runs = 3;  % the run whose time is stored (i_runs<=n_runs)
tol  = 1e-10;

% Generate and solve the five experiments
fprintf('\n\nWhich experiment would you like to run?\n\n');
experiment = 0;

while ( (experiment<=0) | (experiment>5))
  fprintf('\nExperiments');
  fprintf('\n-----------');
  fprintf('\n(1) Experiment 1: B is positive definite--unconstrained case');
  fprintf('\n(2) Experiment 2: B is positive definite--constrained case');
  fprintf('\n(3) Experiment 3: B is singular');
  fprintf('\n(4) Experiment 4: B is indefinite (non-hard case)');
  fprintf('\n(5) Experiment 5: B is indefinite (hard case)');
  fprintf('\n\n\n');
  experiment=input('Enter number: ');
end

fprintf('\n\n');

if (experiment==3) 
  subexperiment = 0;
  while ((subexperiment<1) | (subexperiment>2))
    fprintf('Which subcase would you like to use?\n');
    fprintf('\n(1) Experiment 3a: phi(0) < 0');
    fprintf('\n(2) Experiment 3b: phi(0) >= 0');
    fprintf('\n\n\n');
    subexperiment = input('Enter number: ');
  end
end

if (experiment==4) 
  subexperiment = 0;
  while ((subexperiment<1) | (subexperiment>2))
    fprintf('Which subcase would you like to use?\n');
    fprintf('\n(1) Experiment 4a: Use a randomly generated g');
    fprintf('\n(2) Experiment 4b: A random vector in the orthogonal complement');
    fprintf(' of P||1 and phiBar(-lambda_min)<0');
    fprintf('\n\n\n');
    subexperiment = input('Enter number: ');
  end
end

if (experiment==5) 
  subexperiment = 0;
  while ((subexperiment<1) | (subexperiment>2))
    fprintf('\n(a) Experiment 5a: lambda_min = lambda_1');
    fprintf('\n(b) Experiment 5b: lambda_min = gamma');
    fprintf('\n\n\n');
    subexperiment = input('Enter number: ');
  end
end

for i_outer = 1:size(n,2)  %loop on the value of n

%initialize local variables
gamma = abs(10*randn(1)); 
delta = 1; 

%generate S, Y randomly; ensure Psi has full rank
S = randn(n(i_outer),kp1); 
Y = randn(n(i_outer),kp1);
Psi = Y - gamma.*S;
while rank(Psi)<size(Psi,2)
  S = randn(n(i_outer),kp1); 
  Y = randn(n(i_outer),kp1);
  Psi = Y - gamma.*S;
end
PsiPsi=Psi'*Psi;

g = randn(n(i_outer),1);
  
% compute eigenvalues of $B$
SY          = S'*Y;
SS          = S'*S;
invM        = tril(SY)+ tril(SY,-1)'-gamma.*SS;
invM        = (invM+(invM)')./2;  %symmetrize it

R           = chol(PsiPsi);
RMR         = R* (invM\R'); 
RMR         = (RMR+(RMR)')./2;  %symmetrize it
[U D ]      = eig(RMR);

% Eliminate complex roundoff error then sort eigenvalues and eigenvectors
[D,ind]     = sort(real(diag(D))); 
D           = diag(D);
U           = U(:,ind);            

switch(experiment)

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   case 1
   %%%%%Experiment #1
   %(1) ensure B is strictly positive definite
   %(2) chose delta = mu*\|p_u\|_2, mu = 1.25
   
   %(1) modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
   D           = diag(max(abs(diag(D)),1e-4));  % threshold eigenvalues
   invM        = R'*inv(U*D*U')*R;
   invM        = (invM+(invM)')./2;
   
   %(2) set delta correctly so the unconstrained solution is in trust region
   Psig        = Psi'*g; 
   p_u         = ComputeSBySMW(gamma,g,Psig,Psi,invM,PsiPsi);
   delta       = 1.25*norm(p_u);
   
   % do n_runs runs, save time associated with i_runs
   for testruns = 1:n_runs
     tic
     [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
     timer = toc;
     if testruns==i_runs
       save_time = timer;
     end
   end

   %%%output
   
   if i_outer==1
     fprintf('\n\n\nExperiment #1... \n');
     fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
     fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta| lambda1+sig*');
     fprintf('     sig*   ');
     fprintf('   phi(sig*)      time');
     fprintf('\n-------  -----------  ----------------  ------------ ');
     fprintf('-----------');
     fprintf(' -----------  ----------- ');
   end
   phiBar_check = (1/norm(pStar) ) - (1/delta);
   fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
   fprintf('%8.5e ', spd_check);
   fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);
   
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   case 2

   %%%%%Experiment #2
   %(1) ensure B is strictly positive definite
   %(2) chose delta = mu*\|p_u\|_2, mu = 0.75
  
   %(1) modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
   D           = diag(max(abs(diag(D)),1e-4));  % threshold eigenvalues
   invM        = R'*inv(U*D*U')*R;
   invM        = (invM+(invM)')./2;  %symmetrize
   
   %(2) set delta correctly so the unconstrained solution is outside the trust region
   Psig        = Psi'*g; 
   p_u         = ComputeSBySMW(gamma,g,Psig,Psi,invM,PsiPsi);
   delta       = rand(1)*norm(p_u);
  
   % do n_runs runs, save time associated with i_run-th run
   for testruns = 1:n_runs
     tic
     [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
     timer = toc;
     if testruns==i_runs
       save_time = timer;
     end
   end
  
   %%%output
   if i_outer ==1
     fprintf('\n\n\nExperiment #2... \n');
     fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
     fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
     fprintf('     sig*   ');
     fprintf('   |phi(sig*)|     time');
     fprintf('\n-------  -----------  ----------------  ------------ ');
     fprintf(' -----------');
     fprintf('  -----------  ----------- ');
   end
   phiBar_check = abs((1/norm(pStar) ) - (1/delta));
   fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
   fprintf(' %8.5e   ', spd_check);
   fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);

   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
   case 3
     switch (subexperiment)
   
     case 1       
        %%%%%Experiment #3a
        %(1) Ensure B is positive semidefinite, singular 
        %(2) Ensure phi(0) < 0
        
        %(1) modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
        Dvec        = max(abs(diag(D)),1e-4);        % threshold eigenvalues
        [Dvec, ind3] = sort(Dvec);
        U           = U(:,ind3);
        r           = 2;                             % multiplicity of 0 eigenvalue
        Dvec(1:r)   = -gamma;
        invM        = R'*U*diag(1./Dvec)*U'*R; 
	    invM        = (invM+(invM)')./2;  %symmetrize invM
	
        %(2) set delta so the unconstrained solution is outside the trust region
        P_parallel  = Psi*(R\U);
        g           = g-P_parallel(:,1:r)*(P_parallel(:,1:r)'*g);
  
        Psig        = Psi'*g; 
        Dee         = Dvec+gamma;
        Dee_inv     = zeros(size(Dee));
        Dee_inv(r+1:end)= 1./Dee(r+1:end);
  
        p_u0        = -P_parallel*(Dee_inv.*(P_parallel'*g)) - g./gamma + P_parallel*(P_parallel'*g)./gamma;
        delta       = rand(1)*norm(p_u0);

        % do n_runs runs, save time associated with i_runs-th run
        for testruns = 1:n_runs
            tic
            [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
            timer = toc;
            if testruns==i_runs
                save_time = timer;
            end
        end

        %%%output
        if i_outer ==1
            fprintf('\n\n\nExperiment #3a... \n');
	    fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
            fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
            fprintf('     sig*   ');
            fprintf('   |phi(sig*)|     time');
            fprintf('\n-------  -----------  ----------------  ------------ ');
            fprintf(' -----------');
            fprintf('  -----------  ----------- ');
        end
        phiBar_check = abs((1/norm(pStar) ) - (1/delta));
        fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
        fprintf(' %8.5e   ', spd_check);
        fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);

      case 2
        %%%%%Experiment #3b
        %(1) Ensure B is positive semidefinite, singular 
        %(2) Ensure phi(0) >= 0
        
        %(1) modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
        Dvec        = max(abs(diag(D)),1e-4);        % threshold eigenvalues
        [Dvec, ind3] = sort(Dvec);
        U           = U(:,ind3);
        r           = 2;                             % multiplicity of 0 eigenvalue
        Dvec(1:r)   = -gamma;
        invM        = R'*U*diag(1./Dvec)*U'*R; 
	    invM        = (invM+(invM)')./2;
	
        %(2) set delta so the unconstrained solution is outside the trust region
        P_parallel  = Psi*(R\U);
        g           = g-P_parallel(:,1:r)*(P_parallel(:,1:r)'*g);
  
        Psig        = Psi'*g; 
        Dee         = Dvec+gamma;
        Dee_inv     = zeros(size(Dee));
        Dee_inv(r+1:end)= 1./Dee(r+1:end);
  
        p_u0        = -P_parallel*(Dee_inv.*(P_parallel'*g)) - g./gamma + P_parallel*(P_parallel'*g)./gamma;
        delta       = (1.001+rand(1))*norm(p_u0);


        % do 10 runs, save time associated with 6th run
        for testruns = 1:n_runs
            tic
            [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
            timer = toc;
            if testruns==i_runs
                save_time = timer;
            end
        end

        %%%output
        if i_outer ==1
            fprintf('\n\n\nExperiment #3b... \n');
	    fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
            fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
            fprintf('     sig*   ');
            fprintf('    phi(sig*)      time');
            fprintf('\n-------  -----------  ----------------  ------------ ');
            fprintf(' -----------');
            fprintf('  -----------  ----------- ');
        end
        phiBar_check = (1/norm(pStar) ) - (1/delta);
        fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
        fprintf(' %8.5e   ', spd_check);
        fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);

    end %end of cases 3a and 3b
  
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   case 4
  
     switch (subexperiment)
   
     case 1
     %%%%%Experiment #4a
     %(1) B must be indefinite
     %(2) g is randomly generated
     %(3) phi(-lambda_1) < 0
     
     %(1) modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
     r         = 2;  %multiplicity of leftmost eigenvalue
     D         = max(abs(diag(D)),1e-4);  % threshold eigenvalues
     for j=1:r
       D(j)    = min([-rand(1)-gamma; (-1e-4)-gamma]);        %random value for eigenvalue
     end
     [D, ind3] = sort(D);
     U         = U(:,ind3);       
     invM      = R'*U*diag(1./D)*U'*R;  %modified so that M is still invertible but B is singular
     invM      = (invM+(invM)')./2;  %symmetrize it
     
     % do n_runs runs, save time associated with i_runs-th run
      for testruns = 1:n_runs
	    tic
	    [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
	    timer = toc;
	    if testruns==i_runs
	      save_time = timer;
	    end
      end
      
      %%%output
      if i_outer ==1
	fprintf('\n\n\nExperiment #4a... \n');
	fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
        fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
        fprintf('     sig*   ');
        fprintf('   |phi(sig*)|     time');
        fprintf('\n-------  -----------  ----------------  ------------ ');
        fprintf(' -----------');
        fprintf('  -----------  ----------- ');
      end
      phiBar_check = abs((1/norm(pStar) ) - (1/delta));
      fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
      fprintf(' %8.5e   ', spd_check);
      fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);

    case 2
    %%%%%Experiment #4b
    %(1) B must be indefinite
    %(2) case b: generate g by projecting a vector into P_parallel(:,1) perp
    %(3) phiBar(-\lambda_1)<0
    
    % modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
    D           = max(abs(diag(D)),1e-4);  % threshold eigenvalues
    r = 2;
    D(1:r)      = min([-rand(1)-gamma; (-1e-4)-gamma]);        %random value for eigenvalue
    [D, ind3]   = sort(D);
    U           = U(:,ind3);       
    invM        = R'*U*diag(1./D)*U'*R;  %modified so that M is still invertible but B is singular
    invM        = (invM+(invM)')./2; %symmetrize it
 
    %for the pseudoinverse				     
    Dee         = D+gamma+abs(min(D+gamma)); 
    Dee_inv     = zeros(size(Dee));
    Dee_inv(r+1:end) = 1./Dee(r+1:end);
  
    %(2) project g into orthogonal complement of P_parallel_1
    P_parallel  = Psi*(R\U);
    g           = g-P_parallel(:,1:r)*(P_parallel(:,1:r)'*g);
    Psig        = Psi'*g; 

    %(3) phiBar(-\lambda_1)<0
    p_u0        = -P_parallel*(Dee_inv.*(P_parallel'*g)) - g./(gamma+abs(min(D+gamma))) + P_parallel*(P_parallel'*g)./(gamma+abs(min(D+gamma)));
    delta       = rand(1)*norm(p_u0);				     

    % do n_runs runs, save time associated with i_run-th run
    for testruns = 1:n_runs
      tic
      [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
      timer = toc;
      if testruns==i_runs
	     save_time = timer;
      end
    end
    
    %%%output
    if i_outer ==1
      fprintf('\n\n\nExperiment #4b... \n');
      fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
      fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
      fprintf('     sig*   ');
      fprintf('   |phi(sig*)|     time');
      fprintf('\n-------  -----------  ----------------  ------------ ');
      fprintf(' -----------');
      fprintf('  -----------  ----------- ');
    end
    phiBar_check = abs((1/norm(pStar) ) - (1/delta));
    fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
    fprintf(' %8.5e   ', spd_check);
    fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);
   end  %end of 4a and 4b cases
 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
   case 5
       
    switch (subexperiment)
    case 1     
    %%%%%Experiment #5a
    %(1) B must be indefinite
    %(2) generate g by projecting a vector into P_parallel(:,1) perp
    %(3) phiBar(-\lambda_1)>0
    
    % modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
    D           = max(abs(diag(D)),1e-4);  % threshold eigenvalues
    r = 2;
    D(1:r)      = min([-rand(1)-gamma; (-1e-4)-gamma]);        %random value for eigenvalue
    [D, ind3]   = sort(D);
    U           = U(:,ind3);       
    invM        = R'*U*diag(1./D)*U'*R;  %modified so that M is still invertible but B is singular
    invM        = (invM+(invM)')./2; %symmetrize it

    %help form the pseudoinverse				     
    Dee         = D+gamma+abs(min(D+gamma)); 
    Dee_inv     = zeros(size(Dee));
    Dee_inv(r+1:end) = 1./Dee(r+1:end);
  
    %(2) project g into orthogonal complement of P_parallel_1
    P_parallel  = Psi*(R\U);
    g           = g-P_parallel(:,1:r)*(P_parallel(:,1:r)'*g);
    Psig        = Psi'*g; 

    %(3) phiBar(-\lambda_1)>0
    p_u0        = -P_parallel*(Dee_inv.*(P_parallel'*g)) - g./(gamma+abs(min(D+gamma))) + P_parallel*(P_parallel'*g)./(gamma+abs(min(D+gamma)));
    delta       = (1+rand(1))*norm(p_u0);				     
    norm(MultiplyBVec(p_u0,gamma,Psi,inv(invM))-min(D+gamma)*p_u0+g);

    % do n_runs runs, save time associated with i_run-th run
    for testruns = 1:n_runs
      tic
      [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
      timer = toc;
      if testruns==i_runs
	    save_time = timer;
      end
    end
    
    %%%output
    if i_outer ==1
      fprintf('\n\n\nExperiment #5a... \n');
      fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
      fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|  lambda1+sig*');
      fprintf('     sig*   ');
      fprintf('   |phi(p*)|     time');
      fprintf('\n-------  -----------  ----------------  ------------ ');
      fprintf(' -----------');
      fprintf('  -----------  ----------- ');
    end
    phiBar_check = abs((1/norm(pStar) ) - (1/delta));
    fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
    fprintf(' %8.5e   ', spd_check);
    fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);
 %   fprintf('\nphi(p*) should not be reported...\n');
 
    case 2 
      %%%%%Experiment #5b
      %(1) B must be indefinite
      %(2) generate g by projecting a vector into P_parallel(:,1) perp
      %(3) phiBar(-\lambda_1)>0
     
      % modify eigenvalues of $B$ and re-form invM, leaving Psi unchanged
      D           = max(abs(diag(D)),1e-4);  % threshold eigenvalues
      gamma       = -gamma;
      [D, ind3] = sort(D);
      U           = U(:,ind3);       
      invM        = R'*U*diag(1./D)*U'*R;  %modified so that M is still invertible but B is singular
      invM        = (invM+(invM)')./2; %symmetrize it
 
      %for the pseudoinverse				     
      Dee         = D;
      Dee_inv     = 1./D;
      
      %(2) project g into orthogonal complement of P_parallel_1
      P_parallel  = Psi*(R\U);
      g           = P_parallel*(P_parallel'*g);
      Psig        = Psi'*g; 
      
      %(3) phiBar(-\lambda_1)<0
      p_u0        = -P_parallel*(Dee_inv.*(P_parallel'*g)) - g./(gamma+abs(min(D+gamma))) + P_parallel*(P_parallel'*g)./(gamma+abs(min(D+gamma)));
      delta       = (1+rand(1))*norm(p_u0);				     
      norm(MultiplyBVec(p_u0,gamma,Psi,inv(invM))-gamma*p_u0+g);

      % do n_runs runs, save time associated with i_run-th run
      for testruns = 1:n_runs
	    tic
	    [sigmaStar,pStar,opt1,opt2,spd_check]=obs(g,[ ],[ ],delta,gamma,Psi,invM);
	    timer = toc;
	    if testruns==i_runs
	      save_time = timer;
	    end
      end
    
      %%%output
      if i_outer ==1
        fprintf('\n\n\nExperiment #5b... \n');
	    fprintf('Repeating each problem %d times saving the time of run #%d...\n\n',n_runs,i_runs);
        fprintf('\n   n      ||Bp*+g||   sig*||p*|-delta|   gamma+sig*');
        fprintf('      sig*   ');
        fprintf('    |phi(p*)|     time');
        fprintf('\n-------  -----------  ----------------  ------------ ');
        fprintf(' -----------');
        fprintf('  -----------  ----------- ');
      end
      phiBar_check = abs((1/norm(pStar) ) - (1/delta));
      fprintf('\n%1.1e  %8.5e     %8.5e   ',n(i_outer),opt1,opt2);
      fprintf(' %8.5e   ', gamma+sigmaStar);
      fprintf('%8.5e  %8.5e  %8.5e ' , sigmaStar,phiBar_check,save_time);
     
      otherwise      
    end

 otherwise
end

end  %for loop
fprintf('\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
function [ sStar ] = ComputeSBySMW( tauStar, g, Psig, Psi, invM, PsiPsi)
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
sStar           = -g./tauStar + Psi*(vw\Psig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
function [ Bx ] = MultiplyBVec( x, gamma, Psi, M )
%MultiplyBVec multiplies a vector with the limited memory matrix B
%using the compact representation of B:
% Bx = gammaIx + Psi M Psi'x
%
% Output : 
% Bx := nxn vector.
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Bx      = gamma.*x + Psi*(M*(Psi'*x));

