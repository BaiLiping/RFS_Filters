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
    def getPromisingNewTarget(self, currentParticlesKinematicTmp, currentExistencesTmp, measurements):
        '''
        function [ newIndexes, measurements ] = getPromisingNewTargets( currentParticlesKinematicTmp, currentParticlesExtentTmp, currentExistencesTmp, measurements, parameters )
        numMeasurements = size( measurements, 2 );
        numParticles = size( currentParticlesKinematicTmp, 2 );
        measurementsCovariance = parameters.measurementVariance * eye( 2 );
        surveillanceRegion = parameters.surveillanceRegion;
        areaSize = ( surveillanceRegion( 2, 1 ) - surveillanceRegion( 1, 1 ) ) * ( surveillanceRegion( 2, 2 ) - surveillanceRegion( 1, 2 ) );
        meanMeasurements = parameters.meanMeasurements;
        meanClutter = parameters.meanClutter;
        constantFactor = areaSize * ( meanMeasurements / meanClutter );
        
        probabilitiesNew = ones( numMeasurements, 1 );
        for measurement = 1:numMeasurements
            numTargets = size( currentParticlesKinematicTmp, 3 );
        
            inputDA = ones( 2, numTargets );
            likelihoods = zeros( numParticles, numTargets );
            for target = 1:numTargets
                likelihoods( :, target ) = constantFactor .* exp( getLogWeightsFast( measurements( :, measurement ), currentParticlesKinematicTmp( 1:2, :, target ), getSquare2Fast( currentParticlesExtentTmp( :, :, :, target ) ) + repmat( measurementsCovariance, [ 1, 1, numParticles ] ) ) );
                inputDA( 2, target ) = mean( likelihoods( :, target ), 1 );
                inputDA( :, target ) = currentExistencesTmp( target ) * inputDA( :, target ) + ( 1 - currentExistencesTmp( target ) ) * [ 1;0 ];
            end
        
            inputDA = inputDA( 2, : ) ./ inputDA( 1, : );
            sumInputDA = 1 + sum( inputDA, 2 );
            outputDA = 1 ./ ( repmat( sumInputDA, [ 1, numTargets ] ) - inputDA );
            probabilitiesNew( measurement ) = 1 ./ sumInputDA;
        
            if ( measurement == numMeasurements )
                break ;
            end
        
            for target = 1:numTargets
                logWeights = log( ones( numParticles, 1 ) + likelihoods( :, target ) * outputDA( 1, target ) );
                [ currentParticlesKinematicTmp( :, :, target ), currentParticlesExtentTmp( :, :, :, target ), currentExistencesTmp( target ) ] = updateParticles( currentParticlesKinematicTmp( :, :, target ), currentParticlesExtentTmp( :, :, :, target ), currentExistencesTmp( target ), logWeights, parameters );
            end
        end
        
        [ newIndexes, indexesReordered ] = getCentralReordered( measurements, probabilitiesNew, parameters );
        
        measurements = measurements( :, indexesReordered );
        
        end
        '''

    def getCentralReordered(self,measurements, probabilitiesNew):
        '''
        function [ centralIndexes, indexesReordered ] = getCentralReordered( measurements, probabilitiesNew, parameters )
        threshold = parameters.freeThreshold;
        clusterThreshold = parameters.clusterThreshold;
        meanExtentBirth = ( parameters.priorExtent1 / ( parameters.priorExtent2 - 3 ) ) ^ 2;
        measurementsCovariance = parameters.measurementVariance * eye( 2 ) + meanExtentBirth;
        minClusterElements = parameters.minClusterElements;
        
        allIndexesNumeric = ( 1:size( measurements, 2 ) )';
        
        freeIndexes = probabilitiesNew >= threshold;
        assignedIndexes = probabilitiesNew < threshold;
        
        measurementsFree = measurements( :, freeIndexes );
        
        freeIndexesNumeric = allIndexesNumeric( freeIndexes );
        assignedIndexesNumeric = allIndexesNumeric( assignedIndexes );
        
        clusters = getClusters( measurementsFree, measurementsCovariance, clusterThreshold )';
        
        numElements = sum( clusters > 0, 1 );
        [ numElements, indexes ] = sort( numElements, 'descend' );
        clusters = clusters( :, indexes );
        
        notUsedIndexes = clusters( :, numElements < minClusterElements );
        notUsedIndexes = nonzeros( notUsedIndexes( : ) );
        notUsedIndexesNumeric = freeIndexesNumeric( notUsedIndexes );
        numNotUsed = size( notUsedIndexesNumeric, 1 );
        
        clusters( :, numElements < minClusterElements ) = [  ];
        indexesNumericNew = zeros( 0, 1 );
        numClusters = size( clusters, 2 );
        centralIndexes = zeros( numClusters, 1 );
        for cluster = 1:numClusters
        
            indexes = nonzeros( clusters( :, cluster ) );
            currentMeasurements = measurementsFree( :, indexes );
        
            currentIndexesNumeric = freeIndexesNumeric( indexes );
        
            if ( numel( indexes ) > 1 )
                numMeasurements = size( indexes, 1 );
                distanceMatrix = zeros( numMeasurements, numMeasurements );
                for measurement1 = 1:numMeasurements
                    for measurement2 = ( measurement1 + 1 ):numMeasurements
                        distVector = currentMeasurements( :, measurement1 ) - currentMeasurements( :, measurement2 );
        
                        distanceMatrix( measurement1, measurement2 ) = sqrt( distVector' / measurementsCovariance * distVector );
                        distanceMatrix( measurement2, measurement1 ) = distanceMatrix( measurement1, measurement2 );
                    end
                end
        
                distanceVector = sum( distanceMatrix, 2 );
                [ ~, indexes ] = sort( distanceVector, 'descend' );
                currentIndexesNumeric = currentIndexesNumeric( indexes );
        
            end
            indexesNumericNew = [ currentIndexesNumeric;indexesNumericNew ];
        
            centralIndexes( 1:cluster ) = centralIndexes( 1:cluster ) + numMeasurements;
        end
        
        indexesReordered = [ notUsedIndexesNumeric;indexesNumericNew;assignedIndexesNumeric ];
        
        centralIndexes = centralIndexes + numNotUsed;
        centralIndexes = sort( centralIndexes, 'descend' );
        end
        
        function [ clusters ] = getClusters( measurements, measurementsCovariance, thresholdProbability )
        numMeasurements = size( measurements, 2 );
        
        if ( ~numMeasurements )
            clusters = [  ];
            return ;
        end
        
        thresholdDistance = chi2inv( thresholdProbability, 2 );
        
        distanceVector = zeros( ( numMeasurements * ( numMeasurements - 1 ) / 2 + 1 ), 1 );
        distanceMatrix = zeros( numMeasurements, numMeasurements );
        entry = 1;
        for measurement1 = 1:numMeasurements
            for measurement2 = ( measurement1 + 1 ):numMeasurements
                distVector = measurements( :, measurement1 ) - measurements( :, measurement2 );
        
                entry = entry + 1;
                distanceVector( entry ) = sqrt( distVector' / measurementsCovariance * distVector );
        
                distanceMatrix( measurement1, measurement2 ) = distanceVector( entry );
                distanceMatrix( measurement2, measurement1 ) = distanceVector( entry );
            end
        end
        
        distanceVector = sort( distanceVector );
        distanceVector( distanceVector > thresholdDistance ) = [  ];
        distance = distanceVector( end  );
        
        clusterNumbers = zeros( numMeasurements, 1 );
        clusterId = 1;
        for measurement = 1:numMeasurements
            if ( clusterNumbers( measurement ) == 0 )
                clusterNumbers( measurement ) = clusterId;
                clusterNumbers = findNeighbors( measurement, clusterNumbers, clusterId, distanceMatrix, distance );
                clusterId = clusterId + 1;
            end
        end
        numClusters = clusterId - 1;
        
        maxElements = sum( clusterNumbers == mode( clusterNumbers ) );
        clusters = zeros( 0, maxElements );
        index = 0;
        for cluster = 1:numClusters
            associationTmp = find( clusterNumbers == cluster )';
            numElements = numel( associationTmp );
            if ( numElements <= maxElements )
                index = index + 1;
                clusters( index, : ) = [ zeros( 1, maxElements - numElements ), associationTmp ];
            end
        end
        
        end
        
        function [ cellNumbers ] = findNeighbors( index, cellNumbers, cellId, distanceMatrix, distanceThreshold )
        numMeasurements = size( distanceMatrix, 2 );
        
        for measurement = 1:numMeasurements
            if ( measurement ~= index && distanceMatrix( measurement, index ) < distanceThreshold && cellNumbers( measurement ) == 0 )
                cellNumbers( measurement ) = cellId;
                cellNumbers = findNeighbors( index, cellNumbers, cellId, distanceMatrix, distanceThreshold );
            end
        end
        
        end
        '''

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
        pass


    def getWeightsUnknown(logWeights,oldExistence,skipIndex):
        '''
        function [weights,updatedExistence] = getWeightsUnknown(logWeights,oldExistence,skipIndex)

        if(skipIndex)
            logWeights(:,skipIndex) = zeros(size(logWeights,1),1);
        end
        
        logWeights = sum(logWeights,2);
        
        aliveUpdate = mean(exp(logWeights),1);
        if(isinf(aliveUpdate))
            updatedExistence = 1;
        else
            alive = oldExistence * aliveUpdate;
            dead = (1 - oldExistence);
            updatedExistence = alive / (dead + alive);
        end
        
        weights = exp(logWeights - max(logWeights));
        weights = 1/sum(weights,1) * weights;
        
        end
        
        '''
    def resampleSystematic(weights,numParticles):
        '''
        function indexes = resampleSystematic(weights,numParticles)
        indexes = zeros(numParticles,1);
        cumWeights = cumsum(weights);
        
        grid = zeros(1,numParticles+1);
        grid(1:numParticles) = linspace(0,1-1/numParticles,numParticles) + rand/numParticles;
        grid(numParticles+1) = 1;
        
        i = 1;
        j = 1;
        while( i <= numParticles )
            if( grid(i) < cumWeights(j) )
                indexes(i) = j;
                i = i + 1;
            else
                j = j + 1;
            end
        end
        end
        '''

    def updateParticles(self, oldParticlesKinematic,oldParticlesExtent,oldExistence,logWeights):
        '''
        function [updatedParticlesKinematic,updatedParticlesExtent,updatedExistence] = updateParticles(oldParticlesKinematic,oldParticlesExtent,oldExistence,logWeights,parameters)
        numParticles = parameters.numParticles;
        regularizationDeviation = parameters.regularizationDeviation;
        
        
        logWeights = sum(logWeights,2);
        
        aliveUpdate = mean(exp(logWeights),1);
        if(isinf(aliveUpdate))
            updatedExistence = 1;
        else
            alive = oldExistence*aliveUpdate;
            dead = (1-oldExistence);
            updatedExistence = alive/(dead+alive);
        end
        
        if(updatedExistence ~= 0)
            logWeights = logWeights-max(logWeights);
            weights = exp(logWeights);
            weightsNormalized = 1/sum(weights)*weights;
            
            indexes = resampleSystematic(weightsNormalized,numParticles);
            updatedParticlesKinematic = oldParticlesKinematic(:,indexes);
            updatedParticlesExtent = oldParticlesExtent(:,:,indexes);
            
            updatedParticlesKinematic(1:2,:) = updatedParticlesKinematic(1:2,:) + regularizationDeviation * randn(2,numParticles);
        else
            updatedParticlesKinematic = nan(size(oldParticlesKinematic));
            updatedParticlesExtent = nan(size(oldParticlesExtent));
        end
        end
        '''

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