'''
Software implements pseudocode described in http://arxiv.org/abs/1203.2995 
'''


import numpy as np
import copy
from scipy.stats import multivariate_normal
import math
from util import mvnpdf

"""
PMB Point Target Filter
"""
class SPA_Filter:
    def __init__(self, model, bayesian_filter_type, motion_model_type):
        self.model = model # use generated model which is configured for all parameters used in GM-PHD filter model for tracking the multi-targets.
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)

    """
    Step 1 and 2 of tabel I in [1]: Prediction for new birth targets and existing/surviving targets (Gaussian components).
        -- Beware the predicted RFS is approximated as Poission point process(both new birth and surviving targets are modeled
            as Poission point process), whose intensity function is PHD of summation of "RFS for surviving(existing) targets 
            and RFS for birth targets". And because we always use Gaussian mixture parameterized PHD to approximate the PHD, 
            so we employee Bayesian filter(e.g. Kalman, as now the motion model is just constant velocity(CV) motion model.) 
            to filtering the parameters of Gaussian mixture PHD.
    """
    def predict(self, updatedIntensity):
        '''
        [currentParticlesKinematic,currentExistences,currentParticlesExtent] = performPrediction(currentParticlesKinematic,currentExistences,currentParticlesExtent,scanTime,parameters);    
        currentAlive = currentExistences*exp(-meanMeasurements);
        currentDead = (1-currentExistences);
        currentExistences = currentAlive./(currentDead+currentAlive);
        numTargets = size(currentParticlesKinematic,3);
        numLegacy = numTargets;

        % get indexes of promising new objects 
        [newIndexes,measurements] = getPromisingNewTargets(currentParticlesKinematic,currentParticlesExtent,currentExistences,measurements,parameters);
        numNew = size(newIndexes,1);
        currentLabels = cat(2,currentLabels,[step*ones(1,numNew);newIndexes']);
        '''

        



    def predict_for_initial_step(self):


    def update(self, Z_k, predictedIntensity):
        '''
        % initialize belief propagation (BP) message passing
        newExistences = repmat(meanBirths * exp(-meanMeasurements)/(meanBirths * exp(-meanMeasurements) + 1),[numNew,1]);
        newParticlesKinematic = zeros(4,numParticles,numNew);
        newParticlesExtent = zeros(2,2,numParticles,numNew);
        newWeights = zeros(numParticles,numNew);
        for target = 1:numNew
            proposalMean = measurements(:,newIndexes(target));
            proposalCovariance = 2 * totalCovariance; % strech covariance matrix to make proposal distribution heavier-tailed than target distribution
            
            newParticlesKinematic(1:2,:,target) = proposalMean + sqrtm(proposalCovariance) * randn(2,numParticles);
            newWeights(:,target) = uniformWeight - log(mvnpdf(newParticlesKinematic(1:2,:,target)', proposalMean', proposalCovariance));
            
            newParticlesExtent(:,:,:,target) = iwishrndFastVector(priorExtent1,priorExtent2,numParticles);
        end
        
        currentExistences = cat(1,currentExistences,newExistences);
        currentExistencesExtrinsic = repmat(currentExistences,[1,numMeasurements]);
        
        currentParticlesKinematic = cat(3,currentParticlesKinematic,newParticlesKinematic);
        currentParticlesExtent = cat(4,currentParticlesExtent,newParticlesExtent);
        
        weightsExtrinsic = nan(numParticles,numMeasurements,numLegacy);
        weightsExtrinsicNew = nan(numParticles,numMeasurements,size(newIndexes,1));
        
        likelihood1 = zeros(numParticles,numMeasurements,numTargets);
        likelihoodNew1 = nan(numParticles,numMeasurements,size(newIndexes,1));
        for outer = 1:numOuterIterations
            
            % perform one BP message passing iteration for each measurement
            outputDA = cell(numMeasurements,1);
            targetIndexes = cell(numMeasurements,1);
            for measurement = numMeasurements:-1:1
                inputDA = ones(2,numLegacy);
                
                for target = 1:numLegacy
                    
                    if(outer == 1)
                        likelihood1(:,measurement,target) = constantFactor * exp(getLogWeightsFast(measurements(:,measurement),currentParticlesKinematic(1:2,:,target),getSquare2Fast(currentParticlesExtent(:,:,:,target)) + repmat(measurementsCovariance,[1,1,numParticles])));
                        inputDA(2,target) = currentExistencesExtrinsic(target,measurement) * mean(likelihood1(:,measurement,target),1);
                    else
                        inputDA(2,target) = currentExistencesExtrinsic(target,measurement) * (weightsExtrinsic(:,measurement,target)'*likelihood1(:,measurement,target));
                    end
                    
                    inputDA(1,target) = 1;
                end
                
                targetIndex = numLegacy;
                targetIndexesCurrent = nan(numLegacy,1);
                
                % only new targets with index >= measurement index are connected to measurement
                for target = numMeasurements:-1:measurement
                    
                    if(any(target==newIndexes))
                        targetIndex = targetIndex + 1;
                        targetIndexesCurrent = [targetIndexesCurrent;target];
                        
                        if(outer == 1)
                            weights = exp(newWeights(:,targetIndex-numLegacy));
                            weights = (weights/sum(weights,1))';
                            likelihoodNew1(:,measurement,targetIndex-numLegacy) = constantFactor * exp(getLogWeightsFast(measurements(:,measurement),currentParticlesKinematic(1:2,:,targetIndex),getSquare2Fast(currentParticlesExtent(:,:,:,targetIndex)) + repmat(measurementsCovariance,[1,1,numParticles])));
                            inputDA(2,targetIndex) = currentExistencesExtrinsic(targetIndex,measurement) * (weights*likelihoodNew1(:,measurement,targetIndex-numLegacy));
                        else
                            inputDA(2,targetIndex) = currentExistencesExtrinsic(targetIndex,measurement) *  (weightsExtrinsicNew(:,measurement,targetIndex-numLegacy)'*likelihoodNew1(:,measurement,targetIndex-numLegacy));
                        end
                        inputDA(1,targetIndex) = 1;
                        
                        if(target == measurement)
                            inputDA(1,targetIndex) = 1 - currentExistencesExtrinsic(targetIndex,measurement);
                        end
                    end
                end
               
             targetIndexes{measurement} = targetIndexesCurrent;   
             outputDA{measurement} = dataAssociationBP(inputDA);   
             
            end
            
            
            % perform update step for legacy targets
            for target = 1:numLegacy
                weights = zeros(size(currentParticlesKinematic,2),numMeasurements);
                for measurement = 1:numMeasurements
                    currentWeights = 1 + likelihood1(:,measurement,target) * outputDA{measurement}(1,target);
                    currentWeights = log(currentWeights);
                    weights(:,measurement) = currentWeights;
                end
                
                % calculate extrinsic information for legacy targets (at all except last iteration) and belief (at last iteration)
                if(outer ~= numOuterIterations)
                    for measurement = 1:numMeasurements
                        [weightsExtrinsic(:,measurement,target),currentExistencesExtrinsic(target,measurement)] = getWeightsUnknown(weights,currentExistences(target),measurement);
                    end
                else
                    [currentParticlesKinematic(:,:,target),currentParticlesExtent(:,:,:,target),currentExistences(target)] = updateParticles(currentParticlesKinematic(:,:,target),currentParticlesExtent(:,:,:,target),currentExistences(target),weights,parameters);
                end
            end
            
            % perform update step for new targets
            targetIndex = numLegacy;
            for target = numMeasurements:-1:1
                if(any(target == newIndexes))
                    
                    targetIndex = targetIndex + 1;
                    weights = zeros(size(currentParticlesKinematic,2),numMeasurements+1);
                    weights(:,numMeasurements+1) = newWeights(:,targetIndex-numLegacy);
                    for measurement = 1:target
                        
                        outputTmpDA = outputDA{measurement}(1,targetIndexes{measurement}==target);
                        
                        if(~isinf(outputTmpDA))
                            currentWeights = likelihoodNew1(:,measurement,targetIndex-numLegacy) * outputTmpDA;
                        else
                            currentWeights = likelihoodNew1(:,measurement,targetIndex-numLegacy);
                        end
                        
                        if(measurement ~= target)
                            currentWeights = currentWeights + 1;
                        end
                        currentWeights = log(currentWeights);
                        weights(:,measurement) = currentWeights;
                    end
                    
                    % calculate extrinsic information for new targets (at all except last iteration) or belief (at last iteration)
                    if(outer ~= numOuterIterations)
                        for measurement = 1:target
                            [weightsExtrinsicNew(:,measurement,targetIndex-numLegacy),currentExistencesExtrinsic(targetIndex,measurement)] = getWeightsUnknown(weights,currentExistences(targetIndex),measurement);
                        end
                    else
                        [currentParticlesKinematic(1:2,:,targetIndex),currentParticlesExtent(:,:,:,targetIndex),currentExistences(targetIndex)] = updateParticles(currentParticlesKinematic(1:2,:,targetIndex),currentParticlesExtent(:,:,:,targetIndex),currentExistences(targetIndex),weights,parameters);
                        currentParticlesKinematic(3:4,:,targetIndex) = mvnrnd([0;0],priorVelocityCovariance,numParticles)';
                    end
                end
            end
        end
        
        % perform pruning
        numTargets = size(currentParticlesKinematic,3);
        isRedundant = false(numTargets,1);
        for target = 1:numTargets
            if(currentExistences(target) < thresholdPruning)
                isRedundant(target) = true;
            end
        end
        currentParticlesKinematic = currentParticlesKinematic(:,:,~isRedundant);
        currentParticlesExtent = currentParticlesExtent(:,:,:,~isRedundant);
        currentLabels = currentLabels(:,~isRedundant);
        currentExistences = currentExistences(~isRedundant);
        '''
        pass

    def loopy_belief_propogation(self,Z_k,updatedIntensity):
        '''
        #function [pupd,pnew] = lbp(wupd,wnew)
        #LBP: LOOPY BELIEF PROPAGATION APPROXIMATION OF MARGINAL ASSOCIATION PROBABILITIES
        #Syntax: [pupd,pnew] = lbp(wupd,wnew)
        #Input:
        # wupd(i,j+1) is PDA likelihood for track i/measurment j 
        #  e.g., P_d N(z_jH*mu_i,H*P_i*H'+R)
        # wupd(i,1) is miss likelihood for target i
        #  e.g., (1-P_d)
        # wnew(j) is false alarm/new target intensity for measurement j
        #  e.g., lambda_fa(z_j) + lambda^u(z_j)
        #Output:
        # Estimates of marginal association probabilities in similar format.
        '''        
        if len(updatedIntensity['wupd'])>0:
            length_of_existing_track = len(updatedIntensity['wupd'])
            length_of_measurements_and_miss = len(updatedIntensity['wupd'][0])
            length_of_measurements=length_of_measurements_and_miss-1
    
            #[n,mp1] = size(wupd)
            #m = mp1-1
            # wupd dimensions n x m+1
            # wnew dimensions m x 1
            # pupd, pnew dimensions same
            
            eps_conv_threshold = 1e-8
            
            mu = np.ones((length_of_existing_track,length_of_measurements)) # mu_ba
            mu_old = np.zeros((length_of_existing_track,length_of_measurements))
            nu = np.zeros((length_of_existing_track,length_of_measurements)) # mu_ab
            #nu =[]
            pupd = np.zeros((length_of_existing_track,length_of_measurements_and_miss))
                                 
            
            # Run LBP iteration
            while np.max(abs(mu-mu_old)) > eps_conv_threshold:
              mu_old = mu
              
              for i in range(length_of_existing_track):
                prd=[]
                for j in range(length_of_measurements):
                    prd.append(updatedIntensity['wupd'][i][j+1]*mu[i][j])
                s = updatedIntensity['wupd'][i][0] + sum(prd)

                for j in range(length_of_measurements):
                    nu[i][j]=updatedIntensity['wupd'][i][j+1]/(s-prd[j])         
              
              for k in range(length_of_measurements):
                s = updatedIntensity['wnew'][k] + sum(nu[:,k])
                for z in range(length_of_existing_track):
                    mu[z][k] = 1/(s - nu[z][k])
        
            # Calculate outputs--for existing tracks then for new tracks
            for i in range(length_of_existing_track):
                #for k in range(length_of_measurements):
                #    pred2.append(updatedIntensity['wupd'][i][k+1]*mu[i][k])
                #s = updatedIntensity['wupd'][i][0] + sum(pred2)
                s = updatedIntensity['wupd'][i][0] + np.dot(updatedIntensity['wupd'][i][1:],mu[i])
                pupd[i][0] = updatedIntensity['wupd'][i][0]/s
                for j in range(length_of_measurements):  
                    pupd[i][j+1] = updatedIntensity['wupd'][i][j+1]*mu[i][j]/s
        else:
            pupd=[]
            nu=np.zeros((1,len(Z_k)))
        
        pnew = [] 
        for k in range(len(Z_k)):
            s = updatedIntensity['wnew'][k] + sum(nu[:,k])
            pnew.append(updatedIntensity['wnew'][k]/s)
        return pupd,pnew

    def tomb(self, pupd, pnew, updatedIntensity):
        '''
        TOMB: TOMB ALGOIRHTM FOR FORMING NEW MULTI-BERNOULLI COMPONENTS
        #Syntax: [r,x,P] = tomb(pupd,rupd,xupd,Pupd,pnew,rnew,xnew,Pnew)
        #Input:
        # pupd(i,j+1) is association probability (or estimate) for measurement
        #  j/track i
        # rupd(i,j+1), xupd(:,i,j+1) and Pupd(:,:,i,j+1) give the probability of
        #  existence, state estimate and covariance for this association hypothesis
        # pupd(i,1) is miss probability (or estimate) for target i
        # rupd(i,1), xupd(:,i,1) and Pupd(:,:,i,1) give the probability of
        #  existence, state estimate and covariance for this miss hypothesis
        # pnew(j) is the probability (or estimate thereof) that measurement j does
        #  not associate with any prior component and is therefore a false alarm or
        #  a new target
        # rnew(j), xnew(:,j) and Pnew(:,:,j) give the probability of existence, 
        #  state estimate and covariance for this new target multi-Bernoulli
        #  component
        #Output:
        # r(i), x(:,i) and P(:,:,i) give the probability of existence, state 
        # estimate and covariance for the i-th multi-Bernoulli  component
        ''' 
        
        r_threshold = 1e-4
        r=[]
        x=[]
        #P=np.zeros((length_of_existing_tracks_plus_new_tracks,stateDimensions,stateDimensions))
        P=[]
        if len(pupd)>0:
            # Infer sizes
            length_of_existing_tracks=len(pupd)
            length_of_measurements_and_miss =len(pupd[0])
            stateDimensions = len(updatedIntensity['xnew'][0])
            length_of_measurements = length_of_measurements_and_miss-1
            length_of_existing_tracks_plus_new_tracks = length_of_existing_tracks + length_of_measurements
            
            # Form continuing tracks
            for i in range(length_of_existing_tracks):
                pr=[]
                for j in range(length_of_measurements_and_miss):
                    pr.append(pupd[i][j]*updatedIntensity['rupd'][i][j])
                r.append(sum(pr))
                pr = pr/sum(pr)
                x.append(np.zeros((stateDimensions,1)))
                P.append(np.zeros((stateDimensions,stateDimensions)))
                for k in range(length_of_measurements_and_miss):
                    x[i]+=pr[k]*updatedIntensity['xupd'][i][k]
                
                for k in range(length_of_measurements_and_miss):
                    v = x[i] - updatedIntensity['xupd'][i][k]
                    P[i]+= pr[k]*(updatedIntensity['Pupd'][i][k] + v*np.transpose(v))
                
        else:
            length_of_existing_tracks_plus_new_tracks = len(updatedIntensity['rnew'])
                
        # Form new tracks (already single hypothesis)
        for k in range(len(updatedIntensity['rnew'])):
            r.append(pnew[k]* updatedIntensity['rnew'][k])
            x.append(updatedIntensity['xnew'][k])
            P.append(updatedIntensity['Pnew'][k])
        
        # Truncate tracks with low probability of existence (not shown in algorithm)
        r_extract=[]
        x_extract=[]
        P_extract=[]
        for i in range(length_of_existing_tracks_plus_new_tracks):
            if r[i] > r_threshold:
                r_extract.append(r[i])
                x_extract.append(x[i])
                P_extract.append(P[i])
     
        updatedIntensity['rupd']=r_extract
        updatedIntensity['xupd']=x_extract
        updatedIntensity['Pupd']=P_extract
     
        return r_extract,x_extract,P_extract, updatedIntensity