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

