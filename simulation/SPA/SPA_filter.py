import numpy as np
import copy
from scipy.stats import multivariate_normal
import math
from util import mvnpdf

"""
Sum_of_Product_Algorithm(SPA) Point Target Filter
"""
class SPA_Filter:
    def __init__(self, model, bayesian_filter_type, motion_model_type):
        self.model = model # use generated model which is configured for all parameters used in GM-PHD filter model for tracking the multi-targets.
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)
    '''
    def getLogWeights(measurement,currentParticlesKinematic,currentParticlesExtent):
        numParticles = len(currentParticlesExtent)
        
        allDeterminantes = permute(currentParticlesExtent(1,1,:).*currentParticlesExtent(2,2,:) - currentParticlesExtent(1,2,:).^2,[3,1,2])
        allFactors = np.log(1./(2*math.pi*math.sqrt(allDeterminantes)))
        
        measurementsReptition = [measurement,[1,numParticles])
        
        part2 = ( measurementsReptition - currentParticlesKinematic(1:2,:) )'
        
        # direct calculation of innovation vector times inverted covariance matrix
        tmp = 1./repmat(allDeterminantes,[1,2]) .* (measurementsReptition' - currentParticlesKinematic(1:2,:)')
        part1(:,1) = tmp(:,1) .* squeeze(currentParticlesExtent(2,2,:)) - tmp(:,2) .* squeeze(currentParticlesExtent(2,1,:))
        part1(:,2) = - tmp(:,1) .* squeeze(currentParticlesExtent(1,2,:)) + tmp(:,2) .* squeeze(currentParticlesExtent(1,1,:))
        
        logWeights = allFactors + ( -1/2*(part1(:,1).*part2(:,1) + part1(:,2).*part2(:,2)) )
        
        return logWeights
    '''
    def resampleSystematic(weights,numParticles):
        indexes = np.zeros(numParticles)
        cumWeights = np.cumsum(weights)
        
        grid = np.zeros(numParticles+1)
        for i in range(numParticles+1):
            grid[i]=0+i/numParticles+np.rand/numParticles
      
        i = 1
        j = 1

        while i <= numParticles:
            if grid[i] < cumWeights(j):
                indexes[i] = j
                i = i + 1
            else:
                j = j + 1
        return indexes

    def getPromisingNewTarget(self, currentParticlesKinematicTmp, currentExistencesTmp, measurements):
        numMeasurements = len(measurements)
        numParticles = len(currentParticlesKinematicTmp)
        probabilitiesNew = np.ones(numMeasurements)
        for measurement in range(numMeasurements):
            numTargets = len(currentParticlesKinematicTmp)
        
            inputDA = np.ones((2, numTargets))
            likelihoods = np.zeros( numParticles, numTargets )
            for target in range(numTargets):
                #likelihoods[target] = [self.filter_model['constantFactor'] * np.exp(self.getLogWeightsFast(measurements[measurement], currentParticlesKinematicTmp[target][:2], self.getSquare2Fast(currentParticlesExtentTmp[target])) +  self.filter_model['measurementsCovariance'] for i in range(numParticles)]
                inputDA[1][target]= np.mean(likelihoods[target])
                inputDA[target] = currentExistencesTmp[target] * inputDA( :, target ) + ( 1 - currentExistencesTmp( target ) ) * [ 1;0 ]
        
            inputDA = inputDA( 2, : ) ./ inputDA( 1, : )
            sumInputDA = 1 + sum( inputDA, 2 )
            outputDA = 1 ./ ( repmat( sumInputDA, [ 1, numTargets ] ) - inputDA )
            probabilitiesNew[measurement]= 1/sumInputDA 
                
            for target in range(numTargets):
                logWeights = np.log(np.ones( numParticles) + likelihoods( :, target) * outputDA( 1, target ))
                currentParticlesKinematicTmp[target], currentParticlesExtentTmp[target], currentExistencesTmp[target] = self.updateParticles(currentParticlesKinematicTmp( :, :, target ), currentParticlesExtentTmp( :, :, :, target ), currentExistencesTmp( target ), logWeights, parameters )

        measurements = measurements( :, indexesReordered )
        
        return newIndexes, measurements

    def getWeightsUnknown(logWeights,oldExistence,skipIndex):

        if(skipIndex)
            logWeights(:,skipIndex) = zeros(size(logWeights,1))
        
        logWeights = sum(logWeights,2)
        
        aliveUpdate = np.mean(np.exp(logWeights))
        if(np.isinf(aliveUpdate))
            updatedExistence = 1
        else:
            alive = oldExistence * aliveUpdate
            dead = (1 - oldExistence)
            updatedExistence = alive / (dead + alive)
        
        weights = np.exp(logWeights - max(logWeights))
        weights = 1/sum(weights,1) * weights
        
        weights,updatedExistence
    

    def dataAssociationBP(inputDA):
        '''
        function [outputDA] = dataAssociationBP(inputDA)
        % perform DA
        inputDA = inputDA(2,:)./inputDA(1,:);
        sumInputDA = 1 + sum(inputDA,2);
        outputDA = 1 ./ (repmat(sumInputDA,[1,size(inputDA,2)]) - inputDA);
    
        % make hard DA decision in case outputDA involves NANs
        if(any(isnan(outputDA)))
            outputDA = zeros(size(outputDA));
            [~,index] = max(inputDA(1,:));
            outputDA(index(1)) = 1;
        end
        end
        '''
        pass
        
    def predict(self, currentParticlesKinematic, currentParticlesExtent, measurement):
        numTargets,numParticles,dim = currentParticlesKinematic.shape
        drivingNoiseVariance = pow(self.filter_model['accelerationDeviation'],2)
        survivalProbability = self.filter_model['survivalProbability']
        degreeFreedomPrediction = self.filter_model['degreeFreedomPrediction']

        for target_index in range(numTargets):
            currentParticlesKinematic[target_index]=[self.filter_model['A']*i+self.filter_model['W']*math.sqrt(drivingNoiseVariance)*np.random.normal(size=(numParticles, 2)) for i in currentParticlesKinematic[target_index]]
        
        currentAlive = [i*np.exp(-self.filter_model['meanMeasurements']) for i in currentExistences]
        currentDead = [1-i for i in currentExistences]
        currentExistences = [currentAlive_i/(currentDead_i+currentAlive_i) for currentAlive_i, currentDead_i in zip(currentAlive, currentDead)]
        numTargets = len(currentParticlesKinematic)
        numLegacy = numTargets

        # get indexes of promising new objects 
        newIndexes,measurements = self.getPromisingNewTargets(currentParticlesKinematic,currentParticlesExtent,currentExistences,measurements)
        numNew = len(newIndexes)
        # initialize belief propagation (BP) message passing
        newExistences = [self.filter_model['meanBirths'] * np.exp(-self.filter_model['meanMeasurements'])/(self.filter_model['meanBirths'] * np.exp(-self.filter_model['meanMeasurements']) + 1) for i in range(numNew)]
        
        newParticlesKinematic = np.zeros((4,self.filter_model['numParticles'],numNew))
        newParticlesExtent = np.zeros((2,2,self.filter_model['numParticles'],numNew))
        newWeights = np.zeros(self.filter_model['numParticles'],numNew)
        for target in range(numNew):
            proposalMean = measurements[newIndexes(target)]
            proposalCovariance = 2 * self.filter_model['totalCovariance'] # strech covariance matrix to make proposal distribution heavier-tailed than target distribution
            newParticlesKinematic[target] = proposalMean + sqrtm(proposalCovariance) * np.randn(2,self.filter_model['numParticles'])
            newWeights[target] =  - np.log(mvnpdf(newParticlesKinematic(1:2,:,target)', proposalMean', proposalCovariance))
            newParticlesExtent[target] = self.iwishrndFastVector(self.filter_model['priorExtent1'],self.filter_model['priorExtent2'],self.filter_model['numParticles'])
        numMeasurements=len(measurement)
        numParticles=self.filter_model['numParticles']
        numLegacy = len(currentParticlesKinematic)
        currentExistences = np.hstach(currentExistences,newExistences)
        currentExistencesExtrinsic = [currentExistences for x in range(numMeasurements)]
        
        currentParticlesKinematic = np.hstack(currentParticlesKinematic,newParticlesKinematic)
        currentParticlesExtent = np.hstack(currentParticlesExtent,newParticlesExtent)

        return currentParticlesKinematic,currentExistences,currentParticlesExtent

    def predict_for_initial_step(self,Z_k):
        # the initial step, each measurement would be treated as new track
        # for each track, there are numParticles of particles associated with it
        numMeasurements= len(Z_k)
        numParticles=self.filter_model['numParticles']
        newTracks={}

        for new_target_index in range(numMeasurements):
            # read out the state variables
            proposalMean=Z_k[new_target_index]['translation']
            proposalPositionMean = proposalMean[:2]
            # strech covariance matrix to make proposal distribution heavier-tailed than target distribution
            proposalCovariance = 2 * self.filter_model['totalCovariance'] 
            # add random noise to the proposed mean
            actual_mean = [[proposalMean[0]+math.sqrt(proposalCovariance)*np.random.normal(),proposalMean[1]+math.sqrt(proposalCovariance)*np.random.normal(),proposalMean[2],proposalMean[3]] for i in range(numParticles)]
            # get weight based on the particle position
            newParticlesWeight=[self.filter_model['uniformWeight'] - np.log(mvnpdf(proposalPositionMean, actual_mean[:2], proposalCovariance) for i in range(numParticles))]
            newTracks[new_target_index]={}
            newTracks[new_target_index]['Particles_Weight']=newParticlesWeight
            newTracks[new_target_index]['Particles_Mean']=actual_mean
            # initialize Existences to be one
            newTracks[new_target_index]['Track_Existence']=1
        
        # register the meta information of the tracks
        newTracks['numNewPotentialTracks']=numMeasurements
        newTracks['numLegacyPotentialTracks']=0
        newTracks['max_ID']=numMeasurements

        return newTracks

    def update(self, Z_k, Tracks):
        numParticles=self.filter_model['numParticles']
        numMeasurements= len(Z_k)
        numNewPotentialTracks=numMeasurements
        numLegacyPotentialTracks=Tracks['numLegacyPotentialTracks']

        weightsExtrinsic = np.zeros((numParticles,numMeasurements,numLegacyPotentialTracks))
        weightsExtrinsicNew = np.zeros(numParticles,numMeasurements,numNewPotentialTracks)
        
        likelihood = np.zeros((numParticles,numMeasurements,numLegacyPotentialTracks))
        likelihoodNew = np.zeros(numParticles,numMeasurements,numLegacyPotentialTracks)
        for outer in range(self.filter_model['numOuterIterations']):
            # perform one BP message passing iteration for each measurement
            output_data_association = [[] for i in range(numMeasurements)]
            targetIndexes = {}
            #for measurement = numMeasurements:-1:1:
            for measurement_index in range(numMeasurements):
                input_data_association = {}
                input_data_association['m_idx_equ_t_idx']=[]
                input_data_association['m_idx_not_equ_t_idx']=[]
                for target_index in range(numLegacyPotentialTracks):
                    if outer == 1:
                        # initiate likelihood at the first round
                        likelihood[measurement_index][target_index] = self.filter_model['constantFactor'] * np.exp(self.getLogWeightsFast(Z_k[measurement_index],Tracks[target]['Particles_Mean'][:2]) + self.filter_model['measurementsCovariance'])
                        input_data_association['m_idx_equ_t_idx'].append(currentExistencesExtrinsic[target][measurement] * np.mean(likelihood1[measurement][target]))
                    else:
                        input_data_association['m_idx_equ_t_idx'].append(currentExistencesExtrinsic[target][measurement]* weightsExtrinsic[target][measurement]*likelihood1[target][measurement])
                    input_data_association['m_idx_not_equ_t_idx'].append(1)
                
                # only new targets with index >= measurement index are connected to measurement
                # for target = numMeasurements:-1:measurement
                for target in range(numMeasurements):
                    if target in newIndexes:
                        targetIndex = targetIndex + 1
                        targetIndexesCurrent = [targetIndexesCurrent;target]
                        
                        if outer == 1:
                            weights = np.exp(newWeights[:targetIndex-numLegacy])
                            weights = weights/sum(weights)
                            likelihoodNew[measurement_index][targetIndex-numLegacy]= [constantFactor * np.exp(self.getLogWeightsFast(measurements[measurement],currentParticlesKinematic[targetIndex][:2],self.getSquare2Fast(currentParticlesExtent[targetIndex]) + self.filter_model['measurementsCovariance'])) for i in range(numParticles)]
                            inputDA[1][target] = currentExistencesExtrinsic[targetIndex][measurement] * weights*likelihoodNew1[measurement][targetIndex-numLegacy]
                        else:
                            inputDA[1][target] = currentExistencesExtrinsic[targetIndex][measurement] * weightsExtrinsicNew[measurement][targetIndex-numLegacy]*likelihoodNew1[measurement][targetIndex-numLegacy]
                        inputDA[0][target] = 1
                        
                        if target == measurement:
                            inputDA[1][targetIndex] = 1 - currentExistencesExtrinsic[targetIndex][measurement]

            targetIndexes[measurement] = targetIndexesCurrent   
            output_data_association{measurement} = dataAssociationBP(inputDA);   
            numParticles=self.filter_model['numParticles']
            numMeasurements= len(Z_k)
            numNewPotentialTracks=numMeasurements
            numLegacyPotentialTracks=Tracks['numLegacyPotentialTracks']
            # perform update step for legacy targets
            for legacy_potential_track_index in range(numLegacyPotentialTracks):
                weights = np.zeros(numLegacyPotentialTracks,numMeasurements)
                for measurement_index in range(numMeasurements):
                    currentWeights = 1 + likelihood[measurement_index][legacy_potential_track_index] * output_data_association{measurement}[legacy_potential_track_index]
                    currentWeights = np.log(currentWeights)
                    weights[measurement_index] = currentWeights
                
                # calculate extrinsic information for legacy targets (at all except last iteration) and belief (at last iteration)
                if outer != self.filter_model['numOuterIterations']:
                    for measurement in range(numMeasurements):
                        weightsExtrinsic[measurement][target],currentExistencesExtrinsic[target][measurement] = self.getWeightsUnknown(weights,currentExistences[target],measurement)
                else:
                    currentParticlesKinematic[target],currentParticlesExtent[target],currentExistences[target] = self.updateParticles(currentParticlesKinematic[target],currentParticlesExtent[target],currentExistences[target],weights)
            '''
            regularizationDeviation = self.filter_model['regularizationDeviation']
            logWeights = sum(logWeights,2)
            
            aliveUpdate = np.mean(np.exp(logWeights),1)
            if np.isinf(aliveUpdate):
                updatedExistence = 1
            else:
                alive = oldExistence*aliveUpdate
                dead = (1-oldExistence)
                updatedExistence = alive/(dead+alive)
            
            if updatedExistence != 0:
                logWeights = logWeights-max(logWeights)
                weights = np.exp(logWeights)
                weightsNormalized = 1/sum(weights)*weights
                indexes = self.resampleSystematic(weightsNormalized,numParticles)
                updatedParticlesKinematic = oldParticlesKinematic[indexes]
                updatedParticlesKinematic[:2] = updatedParticlesKinematic[:2] + regularizationDeviation * randn(2,numParticles)
            '''
            # perform update step for new targets
            targetIndex = numLegacy
            #for target = numMeasurements:-1:1
            for target in range(numMeasurements):
                if target in newIndexes:
                    targetIndex = targetIndex + 1
                    weights = np.zeros((len(currentParticlesKinematic),numMeasurements+1))
                    weights[numMeasurements+1] = newWeights[targetIndex-numLegacy]
                    for measurement in range(target):
                        outputTmpDA = outputDA{measurement}(1,targetIndexes{measurement}==target)
                        
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
                    
                    # calculate extrinsic information for new targets (at all except last iteration) or belief (at last iteration)
                    if(outer ~= numOuterIterations)
                        for measurement = 1:target
                            [weightsExtrinsicNew(:,measurement,targetIndex-numLegacy),currentExistencesExtrinsic(targetIndex,measurement)] = getWeightsUnknown(weights,currentExistences(targetIndex),measurement);
                        end
                    else
                        [currentParticlesKinematic(1:2,:,targetIndex),currentParticlesExtent(:,:,:,targetIndex),currentExistences(targetIndex)] = updateParticles(currentParticlesKinematic(1:2,:,targetIndex),currentParticlesExtent(:,:,:,targetIndex),currentExistences(targetIndex),weights,parameters);
                        currentParticlesKinematic(3:4,:,targetIndex) = mvnrnd([0;0],priorVelocityCovariance,numParticles)';
    
            
            # perform pruning
            numTargets = len(currentParticlesKinematic)
            isRedundant = false(numTargets,1);
            for target = 1:numTargets
                if(currentExistences(target) < thresholdPruning)
                    isRedundant(target) = true;
    
            currentParticlesKinematic = currentParticlesKinematic(:,:,~isRedundant);
            currentParticlesExtent = currentParticlesExtent(:,:,:,~isRedundant);
            currentExistences = currentExistences(~isRedundant);
            return currentParticleKinematic, currentExistences, currentParticlesExtent
    