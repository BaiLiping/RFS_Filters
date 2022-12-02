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

    def getLogWeightsFast(measurement,currentParticlesKinematic,currentParticlesExtent):
        numParticles = len(currentParticlesExtent)
        
        allDeterminantes = permute(currentParticlesExtent(1,1,:).*currentParticlesExtent(2,2,:) - currentParticlesExtent(1,2,:).^2,[3,1,2])
        allFactors = np.log(1./(2*math.pi*math.sqrt(allDeterminantes)))
        
        measurementsReptition = repmat(measurement,[1,numParticles])
        
        part2 = ( measurementsReptition - currentParticlesKinematic(1:2,:) )'
        
        # direct calculation of innovation vector times inverted covariance matrix
        tmp = 1./repmat(allDeterminantes,[1,2]) .* (measurementsReptition' - currentParticlesKinematic(1:2,:)')
        part1(:,1) = tmp(:,1) .* squeeze(currentParticlesExtent(2,2,:)) - tmp(:,2) .* squeeze(currentParticlesExtent(2,1,:))
        part1(:,2) = - tmp(:,1) .* squeeze(currentParticlesExtent(1,2,:)) + tmp(:,2) .* squeeze(currentParticlesExtent(1,1,:))
        
        logWeights = allFactors + ( -1/2*(part1(:,1).*part2(:,1) + part1(:,2).*part2(:,2)) )
        
        return logWeights

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

    def updateParticles(self, oldParticlesKinematic,oldParticlesExtent,oldExistence,logWeights):
        numParticles = self.filter_model['numParticles']
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
            updatedParticlesExtent = oldParticlesExtent[indexes]
            updatedParticlesKinematic(1:2,:) = updatedParticlesKinematic(1:2,:) + regularizationDeviation * randn(2,numParticles)
        else:
            updatedParticlesKinematic = np.nan(len(oldParticlesKinematic))
            updatedParticlesExtent = np.nan(len(oldParticlesExtent))

        return updatedParticlesKinematic,updatedParticlesExtent,updatedExistence

    def getPromisingNewTarget(self, currentParticlesKinematicTmp, currentExistencesTmp, measurements):
        numMeasurements = len(measurements)
        numParticles = len(currentParticlesKinematicTmp)
        measurementsCovariance = self.filter_model['measurementVariance'] * np.eye( 2 )
        surveillanceRegion = self.filter_model['surveillanceRegion']
        areaSize = ( surveillanceRegion( 2, 1 ) - surveillanceRegion( 1, 1 ) ) * ( surveillanceRegion( 2, 2 ) - surveillanceRegion( 1, 2 ) )
        meanMeasurements = self.filter_model['meanMeasurements']
        meanClutter = self.filter_model['meanClutter']
        constantFactor = areaSize * ( meanMeasurements / meanClutter )
        
        probabilitiesNew = np.ones(numMeasurements)
        for measurement in range(numMeasurements):
            numTargets = len(currentParticlesKinematicTmp)
        
            inputDA = np.ones((2, numTargets))
            likelihoods = np.zeros( numParticles, numTargets )
            for target in range(numTargets):
                likelihoods[target] = constantFactor .* exp( getLogWeightsFast( measurements( :, measurement ), currentParticlesKinematicTmp( 1:2, :, target ), getSquare2Fast( currentParticlesExtentTmp( :, :, :, target ) ) + repmat( measurementsCovariance, [ 1, 1, numParticles ] ) ) );
                inputDA( 2, target ) = np.mean(likelihoods[target])
                inputDA( :, target ) = currentExistencesTmp[target] * inputDA( :, target ) + ( 1 - currentExistencesTmp( target ) ) * [ 1;0 ]
        
            inputDA = inputDA( 2, : ) ./ inputDA( 1, : )
            sumInputDA = 1 + sum( inputDA, 2 )
            outputDA = 1 ./ ( repmat( sumInputDA, [ 1, numTargets ] ) - inputDA )
            probabilitiesNew( measurement ) = 1 ./ sumInputDA
                
            for target in range(numTargets):
                logWeights = np.log(np.ones( numParticles) + likelihoods( :, target) * outputDA( 1, target ))
                currentParticlesKinematicTmp[target], currentParticlesExtentTmp[target], currentExistencesTmp[target] = updateParticles(currentParticlesKinematicTmp( :, :, target ), currentParticlesExtentTmp( :, :, :, target ), currentExistencesTmp( target ), logWeights, parameters )
        
        newIndexes, indexesReordered = getCentralReordered( measurements, probabilitiesNew, parameters )
        
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
    
    def iwishrndFastVector(parameter1,parameter2,numParticles):
        '''       
        function [a] = iwishrndFastVector(parameter1,parameter2,numParticles)
        d = zeros(size(parameter1));
        d(1,1,:) = sqrt(parameter1(1,1,:));
        d(2,1,:) = parameter1(2,1,:) ./ d(1,1,:);
        d(2,2,:) = sqrt(parameter1(2,2,:) - d(2,1,:).^2);
        
        if(size(parameter1,3) == 1)
            d = repmat(d,[1,1,numParticles]);
        end
        
        r = 2.*randg((parameter2 - [zeros(1,numParticles);ones(1,numParticles)])./2);
        x = sqrt(r);
        x = [x;randn(1,numParticles)];
        
        detX = 1./(x(1,:) .* x (2,:));
        invX = zeros(2,2,numParticles);
        invX(1,1,:) = detX .* x(2,:);
        invX(2,2,:) = detX .* x(1,:);
        invX(1,2,:) = - detX .* x(3,:);
        
        T = zeros(2,2,numParticles);
        T(1,1,:) = d(1,1,:) .* invX(1,1,:) + d(1,2,:) .* invX(2,1,:);
        T(1,2,:) = d(1,1,:) .* invX(1,2,:) + d(1,2,:) .* invX(2,2,:);
        T(2,1,:) = d(2,1,:) .* invX(1,1,:) + d(2,2,:) .* invX(2,1,:);
        T(2,2,:) = d(2,1,:) .* invX(1,2,:) + d(2,2,:) .* invX(2,2,:);
        
        a = zeros(2,2,numParticles);
        a(1,1,:) = T(1,1,:) .* T(1,1,:) + T(1,2,:) .* T(1,2,:);
        a(1,2,:) = T(1,1,:) .* T(2,1,:) + T(1,2,:) .* T(2,2,:);
        a(2,1,:) = T(2,1,:) .* T(1,1,:) + T(2,2,:) .* T(1,2,:);
        a(2,2,:) = T(2,1,:) .* T(2,1,:) + T(2,2,:) .* T(2,2,:);
        '''
        d = np.zeros(len(parameter1))
        a=[]


        return a
        

    def predict(self, currentParticlesKinematic, currentParticlesExtent, measurement):
        currentAlive = [i*np.exp(-self.filter_model['meanMeasurements']) for i in currentExistences]
        currentDead = [1-i for i in currentExistences]
        currentExistences = [currentAlive_i/(currentDead_i+currentAlive_i) for currentAlive_i, currentDead_i in zip(currentAlive, currentDead)]
        numTargets = len(currentParticlesKinematic)
        numLegacy = numTargets

        return currentParticlesKinematic,currentExistences,currentParticlesExtent

    def predict_for_initial_step(self,currentParticlesKinematic, currentParticlesExtent, measurement):
        currentExistences=[1 for i in range(len(measurement))]
        currentAlive = [i*np.exp(-self.filter_model['meanMeasurements']) for i in currentExistences]
        currentDead = [1-i for i in currentExistences]
        currentExistences = [currentAlive_i/(currentDead_i+currentAlive_i) for currentAlive_i, currentDead_i in zip(currentAlive, currentDead)]
        numTargets = len(currentParticlesKinematic)
        numLegacy = numTargets
       
        return currentParticlesKinematic,currentExistences,currentParticlesExtent

    def update(self, measurement, currentParticlesKinematic,currentExistences,currentParticlesExtent):

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
            newWeights[target] = self.filter_model['uniformWeight'] - np.log(mvnpdf(newParticlesKinematic(1:2,:,target)', proposalMean', proposalCovariance))
            newParticlesExtent[target] = self.iwishrndFastVector(self.filter_model['priorExtent1'],self.filter_model['priorExtent2'],self.filter_model['numParticles'])
        numMeasurements=len(measurement)
        numParticles=self.filter_model['numParticles']
        numLegacy = len(currentParticlesKinematic)
        currentExistences = np.hstach(currentExistences,newExistences)
        currentExistencesExtrinsic = [currentExistences for x in range(numMeasurements)]
        
        currentParticlesKinematic = np.hstack(currentParticlesKinematic,newParticlesKinematic)
        currentParticlesExtent = np.hstack(currentParticlesExtent,newParticlesExtent)
        
        weightsExtrinsic = np.nan((numParticles,numMeasurements,numLegacy))
        weightsExtrinsicNew = np.nan(numParticles,numMeasurements,len(newIndexes))
        
        likelihood1 = np.zeros((numParticles,numMeasurements,numTargets))
        likelihoodNew1 = np.nan(numParticles,numMeasurements,len(newIndexes,1))
        for outer in range(self.filter_model['numOuterIterations']):
            # perform one BP message passing iteration for each measurement
            outputDA = [[] for i in range(numMeasurements)]
            targetIndexes = cell(numMeasurements,1)
            #for measurement = numMeasurements:-1:1:
            for measurement in range(numMeasurements):
                inputDA = np.ones(2,numLegacy)
                for target in range(numLegacy):
                    if outer == 1:
                        likelihood1[measurement][target] = constantFactor * np.exp(self.getLogWeightsFast(measurements[measurement],currentParticlesKinematic[target][:2],self.getSquare2Fast(currentParticlesExtent[target]) + self.filter_model['measurementsCovariance'])))
                        inputDA[1][target] = currentExistencesExtrinsic[target][measurement] * np.mean(likelihood1[measurement][target])
                    else:
                        inputDA[1][target]= currentExistencesExtrinsic[target][measurement]* weightsExtrinsic[target][measurement]*likelihood1[target][measurement]
                    inputDA[0][target] = 1
                
                targetIndex = numLegacy
                targetIndexesCurrent = np.nan(numLegacy,1)
                
                # only new targets with index >= measurement index are connected to measurement
                #for target = numMeasurements:-1:measurement
                for target in range(numMeasurements):
                    if target in newIndexes:
                        targetIndex = targetIndex + 1
                        targetIndexesCurrent = [targetIndexesCurrent;target]
                        
                        if outer == 1:
                            weights = np.exp(newWeights[:targetIndex-numLegacy])
                            weights = weights/sum(weights)
                            likelihoodNew1[measurement][targetIndex-numLegacy]= [constantFactor * np.exp(self.getLogWeightsFast(measurements[measurement],currentParticlesKinematic[targetIndex][:2],self.getSquare2Fast(currentParticlesExtent[targetIndex]) + self.filter_model['measurementsCovariance'])) for i in range(numParticles)]
                            inputDA[1][target] = currentExistencesExtrinsic[targetIndex][measurement] * weights*likelihoodNew1[measurement][targetIndex-numLegacy]
                        else:
                            inputDA[1][target] = currentExistencesExtrinsic[targetIndex][measurement] * weightsExtrinsicNew[measurement][targetIndex-numLegacy]*likelihoodNew1[measurement][targetIndex-numLegacy]
                        inputDA[0][target] = 1
                        
                        if target == measurement:
                            inputDA[1][targetIndex] = 1 - currentExistencesExtrinsic[targetIndex][measurement]

             targetIndexes[measurement] = targetIndexesCurrent   
             outputDA{measurement} = dataAssociationBP(inputDA);               
            
            # perform update step for legacy targets
            for target in range(numLegacy):
                weights = np.zeros(len(currentParticlesKinematic),numMeasurements)
                for measurement in range(numMeasurements):
                    currentWeights = 1 + likelihood1[measurement][target] * outputDA{measurement}(1,target)
                    currentWeights = np.log(currentWeights)
                    weights[measurement] = currentWeights
                
                # calculate extrinsic information for legacy targets (at all except last iteration) and belief (at last iteration)
                if outer != self.filter_model['numOuterIterations']:
                    for measurement in range(numMeasurements):
                        [weightsExtrinsic(:,measurement,target),currentExistencesExtrinsic(target,measurement)] = getWeightsUnknown(weights,currentExistences(target),measurement);
                    end
                else
                    [currentParticlesKinematic(:,:,target),currentParticlesExtent(:,:,:,target),currentExistences(target)] = updateParticles(currentParticlesKinematic(:,:,target),currentParticlesExtent(:,:,:,target),currentExistences(target),weights,parameters)
            
            # perform update step for new targets
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
        currentLabels = currentLabels(:,~isRedundant);
        currentExistences = currentExistences(~isRedundant);
