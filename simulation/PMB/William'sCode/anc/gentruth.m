function [model,measlog,xlog] = gentruth(Pd,lfai,numtruth,Pmid,pmi)
%GENTRUTH: GENERATE SIMULATION TRUTH
%Syntax: [model,measlog,xlog] = gentruth(Pd,lfai,numtruth,n,Pmid,pmi)
%Input:
% Pd is the probability of detection
% lfai is the expected number of false alarms per scan
% numtruth is the maximum number of targets (see below)
% Pmid is the covariance of the initial target state (see below)
% pmi is the "case number" determining target birth times (if pmi=1,
%  targets all exist from the beginning, otherwise a new target is born
%  every 10 time steps, and any targets not existing before the mid-point
%  are born at the mid-point)
%Output:
% model is the matlab structure describing the dynamics and measurement
%  models
% measlog{t}(:,j) is the j-th measurement at time t
% xlog{t}(:,i) is the true state of the i-th target at time t
%Note: simulation is initialised at mid-point in time with target states
% randomly drawn from Gaussian distribution with mean 0 and covariance
% Pmid.
%Software implements simulations described in http://arxiv.org/abs/1203.2995 
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

if (pmi == 1), % in this case, targets are present to begin with
  birthtime = zeros(1,numtruth);
else, % in other cases, a new target appears every 10 time steps
  birthtime = 10*(0:numtruth-1);
end;

% Initialise model
T = 1;
model.F = kron(eye(2),[1 T; 0 1]);
model.Q = 0.01*kron(eye(2),[T^3/3 T^2/2; T^2/2 T]);
model.Qc = chol(model.Q);
model.H = kron(eye(2),[1 0]);
model.R = eye(2);
model.Rc = chol(model.R);
model.Pd = Pd;
model.Ps = 0.999;
model.existThresh = 0.8;

% Initialise new target parameter structure
model.xb = zeros(4,1);
model.Pb = diag([100 1 100 1].^2);
model.lambdau = 10; % initially expect 10 targets present (regardless of true number)
volume = 200*200;
model.lambdab = 0.05; % expect one new target to arrive every 20 scans on average
model.lfai = lfai; % expected number of false alarms (integral of lambda_fa)
model.lambda_fa = lfai/volume; % intensity = expected number / state space volume

simlen = 201; % must be odd
midpoint = (simlen+1)/2; 
numfb = midpoint-1;
measlog = cell(simlen,1);
xlog = measlog;

% Initialise at time midpoint and propagate forward and backwards
x = chol(Pmid)'*randn(size(model.F,1),numtruth);
xf = x; xb = x;
measlog{midpoint} = makemeas(x,model);
xlog{midpoint} = x;
for (t = 1:numfb),
  % Run forward and backward simulation process
  xf = model.F*xf + model.Qc'*randn(size(model.F,1),size(x,2));
  xb = model.F\(xb + model.Qc'*randn(size(model.F,1),size(x,2)));
  measlog{midpoint-t} = makemeas(xb(:,midpoint-t>birthtime),model);
  measlog{midpoint+t} = makemeas(xf,model);
  xlog{midpoint-t} = xb(:,midpoint-t>birthtime);
  xlog{midpoint+t} = xf; % note that all targets exist after midpoint
end;

function z = makemeas(x,model)
%Syntax: z = makemeas(x,model)
%Simulates and returns measurement structure for true target state x, 
%model (H,Rc) and false alarm model (init.lfai)
										 
% Generate target measurements (for every target)
z = model.H*x + model.Rc'*randn(size(model.H,1),size(x,2));
% Simulate missed detection process
z = z(:,rand(size(z,2),1) < model.Pd); 
% Generate false alarms (spatially uniform on [-100,100]^2
z = [z, 200*rand(size(model.H,1),poissrnd(model.lfai))-100]; 
% Shuffle order
z = z(:,randperm(size(z,2))); 


