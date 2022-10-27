'''
the code is an adaption of https://github.com/meyer-ucsd/EOT-TSP-21
From paper Scalable Detection and Tracking of Geometric Extended Objects
Notice the code is designed for extended object tracking, but for the nuScenes dataset, the point measurement assumption is adapted.
particle filter implementation is important in order to avoid coalescing.
'''


import numpy as np
import copy
from scipy.stats import multivariate_normal
import math
from trackers.SPA.util import mvnpdf
from utils.utils import compute_birth_rate

"""
SPA Point Target Filter
"""
class SPA_Filter:
    def __init__(self, model, bayesian_filter_type, motion_model_type, classification):
        self.model = model # use generated model which is configured for all parameters used in GM-PHD filter model for tracking the multi-targets.
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)
        self.classification=classification
    """

    original code:
    % Florian Meyer, 2020


    function [newParticles, newExistences, newExtents] = performPrediction( oldParticles, oldExistences, oldExtents, scanTime, parameters )
    [~,numParticles,numTargets] = size(oldParticles);
    drivingNoiseVariance = parameters.accelerationDeviation^2;
    survivalProbability = parameters.survivalProbability;
    degreeFreedomPrediction = parameters.degreeFreedomPrediction;
    
    [A, W] = getTransitionMatrices(scanTime);
    newParticles = oldParticles;
    newExistences = oldExistences;
    newExtents = oldExtents;
    
    for target = 1:numTargets
        oldExtents(:,:,:,target) = oldExtents(:,:,:,target)/degreeFreedomPrediction;
        newExtents(:,:,:,target) = wishrndFastVector(oldExtents(:,:,:,target),degreeFreedomPrediction,numParticles);
    end
    
    for target = 1:numTargets
        newParticles(:,:,target) = A*oldParticles(:,:,target) + W*sqrt(drivingNoiseVariance)*randn(2,numParticles);
        newExistences(target) = survivalProbability*oldExistences(target);
    end
    
    end
    
    function [A, W] = getTransitionMatrices( scanTime )
    A = diag(ones(4,1));
    A(1,3) = scanTime;
    A(2,4) = scanTime;
    
    W = zeros(4,2);
    W(1,1) = 0.5*scanTime^2;
    W(2,2) = 0.5*scanTime^2;
    W(3,1) = scanTime;
    W(4,2) = scanTime;
    end

    """
    def predict(self, Z_k, updatedIntensity, Z_list, birth_rate, lag_time):
        F = np.eye(4, dtype=np.float64)
        I = lag_time*np.eye(2, dtype=np.float64)
        F[0:2, 2:4] = I
        
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, translation and covariance
        r_predict=[]
        x_predict=[]
        P_predict=[]
        lambdau_predict=[]
        xu_predict=[]
        Pu_predict=[]


        crowd_based_birth_rate=compute_birth_rate(Z_k, self.classification)



        for i in range(len(updatedIntensity['lambdau'])):
            lambdau_predict.append(updatedIntensity['lambdau'][i]*self.model['p_S'])
            xu_predict.append(F.dot(updatedIntensity['xu'][i]))
            Pu_predict.append(self.model['Q_k'] + F.dot(updatedIntensity['Pu'][i]).dot(np.transpose(F)))   # Equation (27) in [1].

        for i in range(len(Z_k)):
            lambdau_predict.append(crowd_based_birth_rate[i])
            delta_x = np.random.uniform(-1, 1)
            delta_y = np.random.uniform(-1, 1)
            xu_predict.append(np.array([Z_k[i]['translation'][0]+delta_x, Z_k[i]['translation'][1]+delta_y,Z_k[i]['velocity'][0], Z_k[i]['velocity'][1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as translation value.
            Pu_predict.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.

        # Prediction for surviving(existing) targets. See step 2 in table I in [1].
        existing_element = len(updatedIntensity['rupd']) # Number of Gaussian components after the pruning and merging step at last frame(stands for number of surviving/existing targets).
        for i in range(existing_element): # For each Gaussian components(existing/surviving targets from last frame)
            # Calculate the predicted weights for surviving targets.
            r_predict.append(self.model['p_S']*updatedIntensity['rupd'][i]) # Equation (25) in [1].
            # Calculate the predicted translation, m, and predicted covariance, P, for surviving targets by performing Kalman filter.
            x_predict.append(F.dot(updatedIntensity['xupd'][i]).astype('float64'))  # Equation (26) in [1].
            P_predict.append(self.model['Q_k'] + F.dot(updatedIntensity['Pupd'][i]).dot(np.transpose(F)))   # Equation (27) in [1].
            #if i==len(Z_list):
            #    print('pause')
            #Z_list[i]['translation'][0]=x_predict[i][0][0]
            #Z_list[i]['translation'][1]=x_predict[i][1][0]
            #Z_list[i]['velocity'][0]=x_predict[i][2][0]
            #Z_list[i]['velocity'][1]=x_predict[i][3][0]

        predictedIntensity = {}
        predictedIntensity['r']=r_predict
        predictedIntensity['x']=x_predict
        predictedIntensity['P']=P_predict
        predictedIntensity['lambdau']=[]
        predictedIntensity['xu']=[]
        predictedIntensity['Pu']=[]
        for i in range(len(lambdau_predict)):
            predictedIntensity['lambdau'].append(lambdau_predict[i])
            predictedIntensity['xu'].append(xu_predict[i])
            predictedIntensity['Pu'].append(Pu_predict[i])

        return predictedIntensity, Z_list

    def predict_for_initial_step(self, Z_k,birth_rate):
        w = []  # weight of a Gaussian component
        m = []  # translation of a Gaussian component
        P = []  # Covariance of a gausssian component

        for i in range(len(Z_k)):
            w.append(birth_rate)
            m.append(np.array([Z_k[i]['translation'][0], Z_k[i]['translation'][1],Z_k[i]['velocity'][0], Z_k[i]['velocity'][1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as translation value.
            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.

        predictedIntensity = {}
        predictedIntensity['r']=[]
        predictedIntensity['x']=[]
        predictedIntensity['P']=[]
        predictedIntensity['lambdau']= w
        predictedIntensity['xu']=m
        predictedIntensity['Pu']=P

        return predictedIntensity

    def update(self, Z_k, predictedIntensity, Z_list, id_max,confidence_score,bernoulli_gating):
        '''
        function [lambdau,xu,Pu,wupd,rupd,xupd,Pupd,wnew,rnew,xnew,Pnew] = update(lambdau,xu,Pu,r,x,P,z,model)
        #UPDATE: CONSTRUCT COMPONENTS OF DISTRIBUTION UPDATED WITH MEASUREMENTS
        #Syntax: [lambdau,xu,Pu,wupd,rupd,xupd,Pupd,wnew,rnew,xnew,Pnew] = 
        #          update(lambdau,xu,Pu,r,x,P,z,model)
        #Input:
        # lambdau(k), xu(:,k) and Pu(:,:,k) give the intensity, state estimate and 
        #  covariance for the k-th mixture component of the unknown target Poisson
        #  Point Process (PPP)
        # r(i), x(:,i) and P(:,:,i) give the probability of existence, state 
        #   estimate and covariance for the i-th multi-Bernoulli component (track)
        # z(:,j) is measurement j
        # model is a structure containing parameters of measurement model
        #Output:
        # lambdau(k), xu(:,k) and Pu(:,:,k) give the updated intensity, state 
        #  estimate and covariance for the k-th mixture component of the unknown 
        #  target Poisson Point Process (PPP)
        # wupd(i,j+1) is association likelihood for measurement j/track i
        # rupd(i,j+1), xupd(:,i,j+1) and Pupd(:,:,i,j+1) give the probability of
        #  existence, state estimate and covariance under this association
        #  hypothesis 
        # wupd(i,1) is miss likelihood for target i
        # rupd(i,1), xupd(:,i,1) and Pupd(:,:,i,1) give the probability of
        #  existence, state estimate and covariance under the miss hypothesis
        # wnew(j) is the likelihood that measurement j does not associate with any 
        #  prior component and is therefore a false alarm or a new target
        # rnew(j), xnew(:,j) and Pnew(:,:,j) give the probability of existence, 
        #  state estimate and covariance for this new target multi-Bernoulli
        #  component
        '''
        clutterIntensity = self.model['clutterIntensity']
        number_of_existing_track = len(predictedIntensity['x'])
        number_of_ppp_track = len(predictedIntensity['xu'])

        lambdab_threshold = self.model['poi_thr']

        wupd = np.zeros((number_of_existing_track,len(Z_k)+1))
        sizeupd = np.zeros((number_of_existing_track,len(Z_k)+1, 3))
        elevationupd = np.zeros((number_of_existing_track,len(Z_k)+1))
        rotationupd = np.zeros((number_of_existing_track,len(Z_k)+1,4))
        idupd=[]
        detectionscoreupd=np.zeros((number_of_existing_track,len(Z_k)+1))
        rupd = np.zeros((number_of_existing_track,len(Z_k)+1))
        xupd = np.zeros((number_of_existing_track,len(Z_k)+1,len(self.model['H_k'][0]),1))
        Pupd = np.zeros((number_of_existing_track,len(Z_k)+1,len(self.model['H_k'][0]),len(self.model['H_k'][0])))

        wnew = []
        rnew = []
        xnew = []
        Pnew = []
        sizenew = []
        elevationnew=[]
        rotationnew=[]
        idnew=[]
        detectionscorenew=[]

        lambdau=[]
        xu=[]
        Pu=[]

        eta = []
        S = []
        K = []
        P = []
        invSi_all=[]

        for i in range(number_of_existing_track):
            eta.append(self.model['H_k'].dot(predictedIntensity['x'][i]).astype('float64')) # Calculate predicted measurements.
            S.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['P'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S[i])
            invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            invSi_all.append(invSi)
            K.append(predictedIntensity['P'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P.append(predictedIntensity['P'][i] - K[i].dot(self.model['H_k']).dot(predictedIntensity['P'][i]).astype('float64'))
            
        for i in range(number_of_existing_track):
            # missed detection
            #wupd[i][0]=1 - predictedIntensity['r'][i] + predictedIntensity['r'][i]*(1-self.model['p_D'])
            wupd[i][0]=1 - predictedIntensity['r'][i] + predictedIntensity['r'][i]*(1-Z_list[i]['detection_score'])
            #rupd[i][0]=(predictedIntensity['r'][i]*(1-self.model['p_D'])/wupd[i][0])
            rupd[i][0]=(predictedIntensity['r'][i]*(1-Z_list[i]['detection_score'])/wupd[i][0])
            xupd[i][0]=(predictedIntensity['x'][i])
            Pupd[i][0]=(predictedIntensity['P'][i])
            if i==len(Z_list):
                print('pause')
            sizeupd[i][0]=Z_list[i]['size']
            elevationupd[i][0]=Z_list[i]['translation'][2]
            rotationupd[i][0]=Z_list[i]['rotation']
            idupd.append(Z_list[i]['id'])
            detectionscoreupd[i][0]=Z_list[i]['detection_score']

            for z in range(len(Z_k)): # Iterate over m measurements/detection points
                z_k = copy.deepcopy([[Z_k[z]['translation'][0]],[Z_k[z]['translation'][1]]])
                innovation_residual = np.array([z_k[0][0] - eta[i][0][0],z_k[1][0] - eta[i][1][0]]).reshape(-1,1).astype(np.float64)
                mahalanobis = np.transpose(innovation_residual).dot(invSi_all[i]).dot(innovation_residual)[0][0]
                if mahalanobis<20:
                    mvn_pdf=mvnpdf(z_k, eta[i], S[i])
                    #wupd[i][z+1]=(self.model['p_D'] * predictedIntensity['r'][i] * mvn_pdf)  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                    wupd[i][z+1]=Z_k[z]['detection_score'] * predictedIntensity['r'][i] * mvn_pdf  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]                
                    rupd[i][z+1]=1
                    xupd[i][z+1]=(predictedIntensity['x'][i] + K[i].dot(z_k - eta[i]).astype('float64'))
                    Pupd[i][z+1]=P[i]
                    sizeupd[i][z+1]=Z_k[z]['size']
                    elevationupd[i][z+1]=Z_k[z]['translation'][2]
                    rotationupd[i][z+1]=Z_k[z]['rotation']
                    detectionscoreupd[i][z+1]=Z_k[z]['detection_score']
                else:
                    wupd[i][z+1]=0  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                    #wupd[i][z+1]=(Z_k[z]['detection_score'] * predictedIntensity['r'][i] * mvn_pdf)  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]                
                    rupd[i][z+1]=0
                    xupd[i][z+1]=predictedIntensity['x'][i] 
                    Pupd[i][z+1]=P[i]
                    sizeupd[i][z+1]=Z_k[z]['size']
                    elevationupd[i][z+1]=Z_k[z]['translation'][2]
                    rotationupd[i][z+1]=Z_k[z]['rotation']
                    detectionscoreupd[i][z+1]=Z_k[z]['detection_score']

        for j in range(number_of_ppp_track):
            lambdau.append(predictedIntensity['lambdau'][j]*(1-self.model['p_D']))
            xu.append(predictedIntensity['xu'][j])
            Pu.append(predictedIntensity['Pu'][j])  
        
        eta_new = []
        S_new = []
        K_new = []
        P_new = []
        invS_new_all=[]
        for i in range(number_of_ppp_track):
            eta_new.append(self.model['H_k'].dot(predictedIntensity['xu'][i]).astype('float64')) # Calculate predicted measurements.
            S_new.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['Pu'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S_new[i])
            invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            K_new.append(predictedIntensity['Pu'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            cov_associated_track_updated=predictedIntensity['Pu'][i] - K_new[i].dot(self.model['H_k']).dot(predictedIntensity['Pu'][i]).astype('float64')
            cov_associated_track_updated = 0.5 * (cov_associated_track_updated + np.transpose(cov_associated_track_updated)) 
            P_new.append(cov_associated_track_updated)
            invS_new_all.append(invSi)
        
        for z in range(len(Z_k)): # Iterate over m measurements/detection points
            mean_sum = np.zeros((len(self.model['H_k'][0]),1))
            cov_sum = np.zeros((len(self.model['H_k'][0]),len(self.model['H_k'][0])))
            weight_of_true_detection = 0
            for j in range(number_of_ppp_track):
                z_k = copy.deepcopy([[Z_k[z]['translation'][0]],[Z_k[z]['translation'][1]]])    
                innovation_residual = np.array([z_k[0][0] - eta_new[j][0][0],z_k[1][0] - eta_new[j][1][0]]).reshape(-1,1).astype(np.float64)
                mahalanobis = np.transpose(innovation_residual).dot(invS_new_all[j]).dot(innovation_residual)[0][0]
                if mahalanobis <4:
                    weight_for_track_detection=Z_k[z]['detection_score'] * predictedIntensity['lambdau'][j] * mvnpdf(z_k, eta_new[j], S_new[j])
                    #weight_for_track_detection=self.model['p_D'] * predictedIntensity['lambdau'][j] * mvnpdf(z_k, eta_new[j], S_new[j])
                    weight_of_true_detection+=weight_for_track_detection  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                    #updated_mean=predictedIntensity['xu'][j] + K_new[j].dot(z_k - eta_new[j])
                    updated_mean=predictedIntensity['xu'][j]+K_new[j].dot(innovation_residual)
                    mean_sum+=weight_for_track_detection*(updated_mean).astype('float64')
                    cov_sum += weight_for_track_detection*P_new[j] + weight_for_track_detection*(updated_mean.dot(np.transpose(updated_mean)))
                
            mean_updated = mean_sum/weight_of_true_detection
            cov_updated = cov_sum/weight_of_true_detection - (mean_updated.dot(np.transpose(mean_updated)))
            probability_of_detection = weight_of_true_detection + clutterIntensity
            #e_updated = weight_of_true_detection/probability_of_detection
            e_updated=1

            wnew.append(probability_of_detection)
            if np.isnan(probability_of_detection):
                print('NaN')
            
            rnew.append(e_updated)
            xnew.append(mean_updated)
            Pnew.append(cov_updated)
            sizenew.append(Z_k[z]['size'])
            elevationnew.append(Z_k[z]['translation'][2])
            rotationnew.append(Z_k[z]['rotation'])
            detectionscorenew.append(Z_k[z]['detection_score'])
            id_max+=1
            idnew.append(id_max)
            
 
        # Combine both miss-detected targets and detected targets part of the GM-PHD update.
        updatedIntensity = {}
        updatedIntensity['wupd'] = wupd
        updatedIntensity['rupd'] = rupd
        updatedIntensity['xupd'] = xupd
        updatedIntensity['Pupd'] = Pupd
        updatedIntensity['sizeupd'] = sizeupd
        updatedIntensity['elevationupd'] = elevationupd
        updatedIntensity['rotationupd'] = rotationupd
        updatedIntensity['idupd'] = idupd
        updatedIntensity['detectionscoreupd'] = detectionscoreupd
              
        updatedIntensity['wnew'] = wnew
        updatedIntensity['rnew'] = rnew
        updatedIntensity['xnew'] = xnew
        updatedIntensity['Pnew'] = Pnew
        updatedIntensity['sizenew'] = sizenew
        updatedIntensity['elevationnew'] = elevationnew
        updatedIntensity['rotationnew'] = rotationnew
        updatedIntensity['detectionscorenew'] = detectionscorenew
        updatedIntensity['idnew'] = idnew

        updatedIntensity['lambdau']=[]
        updatedIntensity['xu']=[]
        updatedIntensity['Pu']=[]

        for i in range(len(lambdau)):
            if lambdau[i]>lambdab_threshold:
                updatedIntensity['lambdau'].append(lambdau[i])
                updatedIntensity['xu'].append(xu[i])
                updatedIntensity['Pu'].append(Pu[i])

        return updatedIntensity, id_max

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
            if len(Z_k)>0:
                length_of_existing_track = len(updatedIntensity['wupd'])
                length_of_measurements_and_miss = len(updatedIntensity['wupd'][0])
                length_of_measurements=length_of_measurements_and_miss-1
        
                eps_conv_threshold = 1e-20
                
                mu = np.ones((length_of_existing_track,length_of_measurements)) # mu_ba
                mu_old = np.zeros((length_of_existing_track,length_of_measurements))
                nu = np.zeros((length_of_existing_track,length_of_measurements)) # mu_ab
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
                pupd=np.zeros((len(updatedIntensity['wupd']),1))
                for i in range(len(updatedIntensity['wupd'])):
                    pupd[i][0]=1
                nu=np.zeros((1,len(Z_k)))

        else:
            pupd=[]
            nu=np.zeros((1,len(Z_k)))
        
        pnew = [] 
        for k in range(len(Z_k)):
            s = updatedIntensity['wnew'][k] + sum(nu[:,k])
            pnew.append(updatedIntensity['wnew'][k]/s)
        return pupd,pnew

    def tomb(self, pupd, pnew, updatedIntensity, r_threshold):
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
        
        r_threshold = r_threshold
        r=[]
        x=[]
        translation=[]
        velocity=[]
        size=[]
        detection_score=[]
        rotation=[]
        id=[]
        P=[]
        if len(pupd)>0:
            # Infer sizes
            length_of_existing_tracks=len(pupd)
            length_of_measurements_and_miss =len(pupd[0])
            stateDimensions = 4
            length_of_measurements = length_of_measurements_and_miss-1
            length_of_existing_tracks_plus_new_tracks = length_of_existing_tracks + length_of_measurements
            
            # Form continuing tracks
            for i in range(length_of_existing_tracks):
                pr=[]
                pr_before=[]
                for j in range(length_of_measurements_and_miss):
                    pr_before.append(pupd[i][j]*updatedIntensity['rupd'][i][j])
                
                r.append(sum(pr_before))
                pr = pr_before/sum(pr_before)
                
                x.append(np.zeros((stateDimensions,1)))
                size.append(np.zeros((1,3)))
                rotation.append(np.zeros((1,4)))
                detection_score.append(0)
                translation.append(np.zeros((1,3)))
                velocity.append(np.zeros((1,2)))
                id.append(updatedIntensity['idupd'][i])
                
                
                P.append(np.zeros((stateDimensions,stateDimensions)))
                for k in range(length_of_measurements_and_miss):
                    x[i]+=pr[k]*updatedIntensity['xupd'][i][k]
                    size[i]+=pr[k]*updatedIntensity['sizeupd'][i][k]
                    rotation[i]+=pr[k]*updatedIntensity['rotationupd'][i][k]
                    detection_score[i]+=pr[k]*updatedIntensity['detectionscoreupd'][i]
                    translation[i]+=[pr[k]*updatedIntensity['xupd'][i][k][0][0],pr[k]*updatedIntensity['xupd'][i][k][1][0],pr[k]*updatedIntensity['elevationupd'][i][k]]
                    velocity[i]+=[pr[k]*updatedIntensity['xupd'][i][k][3][0],pr[k]*updatedIntensity['xupd'][i][k][3][0]]                
                for k in range(length_of_measurements_and_miss):       
                    v = x[i] - updatedIntensity['xupd'][i][k]
                    P[i]+= pr[k]*(updatedIntensity['Pupd'][i][k] + v.dot(np.transpose(v)))
                
                
        else:
            length_of_existing_tracks_plus_new_tracks = len(updatedIntensity['rnew'])
                
        # Form new tracks (already single hypothesis)
        for k in range(len(updatedIntensity['rnew'])):
            r.append(pnew[k]* updatedIntensity['rnew'][k])
            x.append(updatedIntensity['xnew'][k])
            P.append(updatedIntensity['Pnew'][k])
            size.append(np.array([updatedIntensity['sizenew'][k]]))
            rotation.append(np.array([updatedIntensity['rotationnew'][k]]))
            detection_score.append(np.array([updatedIntensity['detectionscorenew'][k]]))
            id.append(updatedIntensity['idnew'][k])
            translation.append(np.array([[updatedIntensity['xnew'][k][0][0],updatedIntensity['xnew'][k][1][0],updatedIntensity['elevationnew'][k]]]))
            velocity.append(np.array([[updatedIntensity['xnew'][k][3][0],updatedIntensity['xnew'][k][3][0]]]))               

        # Truncate tracks with low probability of existence (not shown in algorithm)
        r_extract=[]
        x_extract=[]
        P_extract=[]
        id_extract=[]
        Z_list_new=[]
        for i in range(length_of_existing_tracks_plus_new_tracks):
            if r[i] > r_threshold:
                r_extract.append(r[i])
                x_extract.append(x[i])
                P_extract.append(P[i])
                id_extract.append(id[i])
                element={}
                element['translation']=translation[i][0]
                element['rotation']=rotation[i][0]
                element['velocity']=velocity[i][0]
                element['detection_score']=detection_score[i][0]
                element['id']=id[i]
                element['size']=size[i][0]
                Z_list_new.append(element)

        updatedIntensity['rupd']=r_extract
        updatedIntensity['xupd']=x_extract
        updatedIntensity['Pupd']=P_extract
        updatedIntensity['idupd'] = id_extract
        if len(r_extract)!=len(Z_list_new):
            print('pause')
     
        return r_extract,updatedIntensity, Z_list_new