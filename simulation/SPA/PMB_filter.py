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
class PMB_Filter:
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
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        r_predict=[]
        x_predict=[]
        P_predict=[]
        lambdau_predict=[]
        xu_predict=[]
        Pu_predict=[]

        v_init = [0.0, 0.0] # initial velocity
        w_birthsum = self.model['w_birthsum']
        z_init = self.model['z_init'][0]
        n_birth = self.model['n_birth']
        lambdab_threshold = 1e-3


        for i in range(len(updatedIntensity['lambdau'])):
            lambdau_predict.append(updatedIntensity['lambdau'][i]*self.model['p_S'])
            xu_predict.append(self.model['F_k'].dot(updatedIntensity['xu'][i]))
            Pu_predict.append(self.model['Q_k'] + self.model['F_k'].dot(updatedIntensity['Pu'][i]).dot(np.transpose(self.model['F_k'])))   # Equation (27) in [1].

        
        for i in range(n_birth):
            lambdau_predict.append(w_birthsum)
            xu_predict.append(np.array([z_init[0], z_init[1], v_init[0], v_init[1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            Pu_predict.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.

        # Prediction for surviving(existing) targets. See step 2 in table I in [1].
        numTargets_Jk_minus_1 = len(updatedIntensity['rupd']) # Number of Gaussian components after the pruning and merging step at last frame(stands for number of surviving/existing targets).
        for i in range(numTargets_Jk_minus_1): # For each Gaussian components(existing/surviving targets from last frame)
            # Calculate the predicted weights for surviving targets.
            r_predict.append(self.model['p_S']*updatedIntensity['rupd'][i]) # Equation (25) in [1].
            
            # Calculate the predicted mean, m, and predicted covariance, P, for surviving targets by performing Kalman filter.
            x_predict.append(self.model['F_k'].dot(updatedIntensity['xupd'][i]).astype('float64'))  # Equation (26) in [1].
            P_predict.append(self.model['Q_k'] + self.model['F_k'].dot(updatedIntensity['Pupd'][i]).dot(np.transpose(self.model['F_k'])))   # Equation (27) in [1].
    
        predictedIntensity = {}
        predictedIntensity['r']=r_predict
        predictedIntensity['x']=x_predict
        predictedIntensity['P']=P_predict
        predictedIntensity['lambdau']=[]
        predictedIntensity['xu']=[]
        predictedIntensity['Pu']=[]
        for i in range(len(lambdau_predict)):
            if lambdau_predict[i]>lambdab_threshold:
                predictedIntensity['lambdau'].append(lambdau_predict[i])
                predictedIntensity['xu'].append(xu_predict[i])
                predictedIntensity['Pu'].append(Pu_predict[i])

        return predictedIntensity

    def predict_for_initial_step(self):
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        v_init = [0.0, 0.0] # initial velocity
        w_birthsuminit = self.model['w_birthsuminit']
        n_birthinit = self.model['n_birthinit']
        z_init = self.model['z_init']

        for i in range(n_birthinit):
            w.append(w_birthsuminit/n_birthinit)
            #m.append(np.array([z_k[0], z_k[1], v_init[0], v_init[1]]).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.

            #need to specify dtype=object for it to work on Linux, no need for Win10
            m.append(np.array([z_init[i][0], z_init[i][1], v_init[0], v_init[1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.

        predictedIntensity = {}
        predictedIntensity['r']=[]
        predictedIntensity['x']=[]
        predictedIntensity['P']=[]
        predictedIntensity['lambdau']= w
        predictedIntensity['xu']=m
        predictedIntensity['Pu']=P

        return predictedIntensity

    def update(self, Z_k, predictedIntensity):
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


        lambdab_threshold = 1e-4

        wupd = np.zeros((number_of_existing_track,len(Z_k)+1))
        rupd = np.zeros((number_of_existing_track,len(Z_k)+1))
        xupd = np.zeros((number_of_existing_track,len(Z_k)+1,len(self.model['H_k'][0]),1))
        Pupd = np.zeros((number_of_existing_track,len(Z_k)+1,len(self.model['H_k'][0]),len(self.model['H_k'][0])))
        
        wnew = []
        rnew = []
        xnew = []
        Pnew = []

        lambdau=[]
        xu=[]
        Pu=[]

        eta = []
        S = []
        K = []
        P = []

        for i in range(number_of_existing_track):
            eta.append(self.model['H_k'].dot(predictedIntensity['x'][i]).astype('float64')) # Calculate predicted measurements.
            S.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['P'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S[i])
            invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            K.append(predictedIntensity['P'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P.append(predictedIntensity['P'][i] - K[i].dot(self.model['H_k']).dot(predictedIntensity['P'][i]).astype('float64'))

        for i in range(number_of_existing_track):
            # missed detection
            wupd[i][0]=1 - predictedIntensity['r'][i] + predictedIntensity['r'][i]*(1-self.model['p_D'])
            rupd[i][0]=(predictedIntensity['r'][i]*(1-self.model['p_D'])/wupd[i][0])
            xupd[i][0]=(predictedIntensity['x'][i])
            Pupd[i][0]=(predictedIntensity['P'][i])

            for z in range(len(Z_k)): # Iterate over m measurements/detection points
                z_k = copy.deepcopy(Z_k[z])
                mvn_pdf=mvnpdf(z_k, eta[i], S[i])
                wupd[i][z+1]=(self.model['p_D'] * predictedIntensity['r'][i] * mvn_pdf)  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                rupd[i][z+1]=(1)
                xupd[i][z+1]=(predictedIntensity['x'][i] + K[i].dot(z_k - eta[i]).astype('float64'))
                Pupd[i][z+1]=P[i]

        for j in range(number_of_ppp_track):
            lambdau.append(predictedIntensity['lambdau'][j]*(1-self.model['p_D']))
            xu.append(predictedIntensity['xu'][j])
            Pu.append(predictedIntensity['Pu'][j])  
        
        eta_new = []
        S_new = []
        K_new = []
        P_new = []
        for i in range(number_of_ppp_track):
            eta_new.append(self.model['H_k'].dot(predictedIntensity['xu'][i]).astype('float64')) # Calculate predicted measurements.
            S_new.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['Pu'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S_new[i])
            invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            K_new.append(predictedIntensity['Pu'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P_new.append(predictedIntensity['Pu'][i] - K_new[i].dot(self.model['H_k']).dot(predictedIntensity['Pu'][i]).astype('float64'))

        for z in range(len(Z_k)): # Iterate over m measurements/detection points
            mean_sum = np.zeros((len(self.model['H_k'][0]),1))
            cov_sum = np.zeros((len(self.model['H_k'][0]),len(self.model['H_k'][0])))
            weight_of_true_detection = 0
            for j in range(number_of_ppp_track):
                z_k = copy.deepcopy(Z_k[z])               
                weight_for_track_detection=self.model['p_D'] * predictedIntensity['lambdau'][j] * mvnpdf(z_k, eta_new[j], S_new[j])
                weight_of_true_detection+=weight_for_track_detection  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                updated_mean=predictedIntensity['xu'][j] + K_new[j].dot(z_k - eta_new[j])
                mean_sum+=weight_for_track_detection*(updated_mean).astype('float64')
                cov_sum += weight_for_track_detection*P_new[j] + weight_for_track_detection*(updated_mean.dot(np.transpose(updated_mean)))

            mean_updated = mean_sum/weight_of_true_detection
            cov_updated = cov_sum/weight_of_true_detection - (mean_updated*np.transpose(mean_updated))
            probability_of_detection = weight_of_true_detection + clutterIntensity
            e_updated = weight_of_true_detection/probability_of_detection

            wnew.append(probability_of_detection)
            rnew.append(e_updated)
            xnew.append(mean_updated)
            Pnew.append(cov_updated)
 
        # Combine both miss-detected targets and detected targets part of the GM-PHD update.
        updatedIntensity = {}
        updatedIntensity['wupd'] = wupd
        updatedIntensity['rupd'] = rupd
        updatedIntensity['xupd'] = xupd
        updatedIntensity['Pupd'] = Pupd
              
        updatedIntensity['wnew'] = wnew
        updatedIntensity['rnew'] = rnew
        updatedIntensity['xnew'] = xnew
        updatedIntensity['Pnew'] = Pnew

        updatedIntensity['lambdau']=[]
        updatedIntensity['xu']=[]
        updatedIntensity['Pu']=[]

        for i in range(len(lambdau)):
            if lambdau[i]>lambdab_threshold:
                updatedIntensity['lambdau'].append(lambdau[i])
                updatedIntensity['xu'].append(xu[i])
                updatedIntensity['Pu'].append(Pu[i])

        return updatedIntensity

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