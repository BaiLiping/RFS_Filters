function [r,x,P] = momb(pupd,rupd,xupd,Pupd,pnew,rnew,xnew,Pnew)
%MOMB: MOMB ALGORITHM FOR FORMING NEW MULTI-BERNOULLI COMPONENTS
%Syntax: [r,x,P] = momb(pupd,rupd,xupd,Pupd,pnew,rnew,xnew,Pnew)
%Input:
% pupd(i,j+1) is association probability (or estimate) for measurement
%  j/track i
% rupd(i,j+1), xupd(:,i,j+1) and Pupd(:,:,i,j+1) give the probability of
%  existence, state estimate and covariance for this association hypothesis
% pupd(i,1) is miss probability (or estimate) for target i
% rupd(i,1), xupd(:,i,1) and Pupd(:,:,i,1) give the probability of
%  existence, state estimate and covariance for this miss hypothesis
% pnew(j) is the probability (or estimate thereof) that measurement j does
%  not associate with any prior component and is therefore a false alarm or
%  a new target
% rnew(j), xnew(:,j) and Pnew(:,:,j) give the probability of existence, 
%  state estimate and covariance for this new target multi-Bernoulli
%  component
%Output:
% r(i), x(:,i) and P(:,:,i) give the probability of existence, state 
% estimate and covariance for the i-th multi-Bernoulli  component
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

r_threshold = 1e-4;

% Infer sizes
[nold,mp1] = size(pupd);
stateDimensions = size(xnew,1);
m = mp1-1;
n = nold + m;

% Allocate memory
r = zeros(n,1);
x = zeros(stateDimensions,n);
P = zeros(stateDimensions,stateDimensions,n);

% Generate legacy (missed detection) tracks
r(1:nold) = pupd(:,1) .* rupd(:,1);
x(:,1:nold) = xupd(:,:,1);
P(:,:,1:nold) = Pupd(:,:,:,1);

% Generate updated tracks for each measurement
for (j = 1:m),
  i = j + nold;
  prnew = pnew(j)*rnew(j);
  pr = pupd(:,j+1).*rupd(:,j+1);
  r(i) = sum(pr) + prnew;
  pr = pr/r(i); prnew = prnew/r(i);
  x(:,i) = xupd(:,:,j+1)*pr + xnew(:,j)*prnew;
  v = x(:,i) - xnew(:,j);
  P(:,:,i) = prnew*(Pnew(:,:,j) + v*v');
  for (i2 = 1:nold),
    v = x(:,i) - xupd(:,i2,j+1);
    P(:,:,i) = P(:,:,i) + pr(i2)*(Pupd(:,:,i2,j+1) + v*v');
  end;
end;

% Truncate tracks with low probability of existence (not shown in algorithm)
ss = r > r_threshold;
r = r(ss);
x = x(:,ss);
P = P(:,:,ss);
