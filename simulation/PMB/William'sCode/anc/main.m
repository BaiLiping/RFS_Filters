%MAIN: MAIN PROCESSING LOOP
%Software implements pseudocode described in http://arxiv.org/abs/1203.2995 
% (accepted for publication, IEEE Transactions on Aerospace and Electronic 
% Systems)
%Copyright 2012 Defence Science and Technology Organisation
%Note:
% As this software is provided free charge and to the full extent permitted
% by applicable law the software is provided "as is" and DSTO and any
% copyright holder in material included in the software make no
% representation and gives no warranty of any kind, either expressed or
% implied, including, but not limited to, implied warranties of
% merchantability, fitness for a particular purpose, non-infringement, or
% the presence or absence of defects or errors, whether discoverable or
% not. The entire risk as to the quality and performance of the software is
% with you.  

% Revision log:
% 24Aug16 corrected typo on line 75 (previously would not estimate odd cardinality)

% Select algorithm
alg = 'TOMB';
%alg = 'MOMB';

% Set simulation parameters
Pd = 0.5; % probability of detection
lfai = 10; % expected number of false alarms per scan
numtruth = 6; % number of targets
simcasenum = 1; % simulation case 1 or 2 (see paper)
%simcasenum = 2;
if (simcasenum == 1), % covariance used for mid-point initialisation
  Pmid = 1e-6*eye(4);
else,
  Pmid = 0.25*eye(4);
end;

% Generate truth
[model,measlog,xlog] = gentruth(Pd,lfai,numtruth,Pmid,simcasenum);

% Initialise filter parameters
stateDimensions = size(model.xb,1);
% Multi-Bernoulli representation
n = 0; 
r = zeros(n,1);
x = zeros(stateDimensions,n);
P = zeros(stateDimensions,stateDimensions,n);
% Unknown target PPP parameters
lambdau = model.lambdau;
xu = model.xb;
Pu = model.Pb;

% Loop through time
numTime = length(measlog);
figure(1);clf;
for (t = 1:numTime),
  % Predict
  [r,x,P,lambdau,xu,Pu] = predict(r,x,P,lambdau,xu,Pu,model);
  
  % Update
  [lambdau,xu,Pu,wupd,rupd,xupd,Pupd,wnew,rnew,xnew,Pnew] = ...
    update(lambdau,xu,Pu,r,x,P,measlog{t},model);
  
  % Use LBP to estimate association probabilities
  [pupd,pnew] = lbp(wupd,wnew);
  
  % Form new multi-Bernoulli components using either TOMB or MOMB algorithm
  if (strcmp(alg,'TOMB')),
    % TOMB
    [r,x,P] = tomb(pupd,rupd,xupd,Pupd,pnew,rnew,xnew,Pnew);
    ss = r > model.existThresh;
    n = sum(ss);
  elseif (strcmp(alg,'MOMB')),
    % MOMB
    [r,x,P] = momb(pupd,rupd,xupd,Pupd,pnew,rnew,xnew,Pnew);
    ss = false(size(r));
    if (~isempty(r)),
      pcard = prod(1-r)*poly(-r./(1-r));
      [~,n] = max(pcard);
      [~,o] = sort(-r);
      n = n - 1;
      ss(o(1:n)) = true;
    end;
  else,
    error(['unknown algorithm ' alg]);
  end;
  
  % Display result
  plot(xlog{t}(1,:),xlog{t}(3,:),'k^',...
    measlog{t}(1,:),measlog{t}(2,:),'mx',x(1,ss),x(3,ss),'b*'); 
  axis([-100 100 -100 100]);
  axis square;
  if (t <= 20), % Slow to draw--only draw for first 20 time steps
    if (n == 0),
      legend({'True target','Measurement'},'Location','NorthEastOutside'); 
    else,
      legend({'True target','Measurement','Estimate'},'Location','NorthEastOutside'); 
    end;
  end;
  title(sprintf('%s; number of MB components: %d; birth PPP mixture components: %d\nEstimated number of targets: %d',...
    alg,length(r),length(lambdau),n));
  drawnow;
end;
