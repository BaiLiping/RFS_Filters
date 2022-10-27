"""
%% ------------------------------ Gaussian Mixture(GM) Cardinalized Probability Hypothesis Density(CPHD) filter ------------------------------ %%
This Python code is reproduction for the "point target GM-CPHD filter" originally proposed in paper [2], with assumption
of no target spawning. The code is adapted from "GM-CPHD" matlab code available from Garcia at: https://github.com/Agarciafernandez/MTT.

The key idea behind GM-PHD is that when the SNR is large enough, the first statistical moment(PHD) of RFS multi-target density
is used to approximate the multi-target density during recursive propogation process.

As described in Secion II of [2], a major weakness of GM-PHD process is how the states are extracted. When the cardinality 
informaition is unavailable, all we can do is to set a threshold for weights such that the Gaussian components(single target
probability density) with weights below that threshold is ignored. 

The CPHD filter steers a middle ground between the information loss of first-order multitarget-moment approximation and the 
intractability of a full second-order approximation. Cardinalized PHD try to solve it by propogating a cardinality distribution 
along side the first statistical moment of multi-target density.
%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
Processing
% [2] 2006 B.-N. Vo, Anthonio Cantoni "The Cardinalized Probability Hypothesis Density Filter for Linear Gaussian Multi-Target Models", 
2006 Annual Conference on Information Sciences and Systems (CISS)
% [3] 2007 R. Mahler "PHD filters of higher order in target number" IEEE Transactions on Aerospace and Electronic Systems
% [4] 2003 R. Mahler "Multitarget Bayes Filtering via First-Order Multitarget Moments" IEEE Transactions on Aerospace and Electronic Systems
"""

from matplotlib.pyplot import thetagrids
import numpy as np
import copy
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from scipy.stats import multivariate_normal
from scipy.stats import poisson
import math
from util import mvnpdf, compute_elementary_symmetric_functions,compute_measurement_likelihood_matrix
import random

"""
GM-CPHD Point Target Filter
"""
class GM_CPHD_Filter:
    def __init__(self, model, bayesian_filter_type, motion_model_type):
        self.model = model # use generated model which is configured for all parameters used in GM-PHD filter model for tracking the multi-targets.
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)
    """
    STEP 1: Prediction
    """

    def cardinality_predict(self, updated_cardinality):
        '''
        This function implements the equation (13) in [2], which computes predicted cardinality distribution.

        The entire predicted cardinality distribution is the convolution of "cardinality distribution of new bith targets"
        and "cardinality distribution of surviving targets". 
        For cardinality distribution of surviving targets, 
        For cardinality distribution of new bith targets, since new birth targets are modelled by Poisson Point Process(PPP) 
        RFS, by definition the cardinality of PPP follows Poisson distribution thus cardinality distribution of new bith 
        targets follows Poisson distribution. So the cardinality distribution of new bith targets is represented by using 
        probability mass function(pmf) of Poisson distribution.
        See details of pmf of Poisson distribution here: https://en.wikipedia.org/wiki/Poisson_distribution.

        For more detailed understanding of this function please refer to README.

        Please note that equation (13) of [2] is the final format of predicted cardinality distribution, cardinality_pred, 
        which is the convolution of cardinality_birth and cardinality_survival as we mentioned above. For full derivation 
        of every term of equation (13), one should refer to [3], specifically equation (53)-(56).

        Input:
            cardinality_distribution: The cardinality distribution after update step. It is a list of length n_max. The value 
                of every element in this list represents the probability(how possible) the cardinality is "index of that element + 1".
        Return:
            cardinality_pred: The predicted cardinality distribution. The format is same as cardinality_distribution.
        '''
        cardinality_survive = np.zeros(self.model['n_max']+1) # Initiate the survive cardinality as a list with length n_max, the maximum number of targets per frame.
        cardinality_pred    = np.zeros(self.model['n_max']+1) # Initiate the survive cardinality as a list with length n_max, the maximum number of targets per frame.
        cardinality_birth   = np.zeros(self.model['n_max']+1) # Initiate a list with length n_max, the maximum number of targets per frame.
        # Calculate cardinality distribution of new birth targets. It is the first part(the part before the second summation) of equation (13) in [2] with changing
        # the (n - j) to j.
        intensity_lambda =self.model['w_birthsum']
        for n in range(self.model['n_max']+1):
            # PPP RFS distribution's set integral is Poisson Distribution, which is why the cardinality distribution of PPP(new birth targets) is a poisson distribution.
            # Thus here we calculate cardinality distribution of new birth targets by using pmf of poission distribution.
            # To refresh poisson distribution pmf, refer to: https://en.wikipedia.org/wiki/Poisson_distribution.
            cardinality_birth[n]=(intensity_lambda**n) * np.exp(-intensity_lambda)/math.factorial(n)
        
        # Calculate cardinality distribution of surviving targets. This part is calculated according to PGFL(probability generating functional) based on updated 
        # cardinality distribution at last frame. It is the second part(start from the second summation) of equation (13) in [2].
        for n in range(self.model['n_max']+1):
            for p in range(self.model['n_max']+1)[n:]:
                cardinality_survive[n]+=np.exp(sum([np.log(x+1) for x in range(p)])
                                               -sum([np.log(x+1) for x in range(n)])
                                               -sum([np.log(x+1) for x in range(p-n)])
                                               +n*np.log(self.model['p_S'])
                                               +(p-n)*np.log(1-self.model['p_S'])
                                               )*updated_cardinality[p]
  
        # Calculate predicted cardinality, which is the convolution of birth cardinality distribution, cardinality_birth, and surviving cardinality distribution, cardinality_survival.
        # It is exact the equation (13) in [2].
        for n in range(self.model['n_max']+1):
            # To get the distribution of new set of "A union B", convolution is required.
            cardinality_pred[n]=sum(x*y for x, y in zip(cardinality_survive[:n+1], np.flip(cardinality_birth[:n+1])))
        # Normalize predicted cardinality distribution.
        cardinality_pred = cardinality_pred/sum(cardinality_pred)

        return cardinality_pred

    def cardinality_predict_initial_step(self):
        '''
        Compute the predicted cardinality distribution for the initial step.
        This one only has the birth part of predicted cardinality distribution, cardinality_predict(Due to no surviving targets at initial step).
        Also, the intensity used here is w_birthinit instead of w_birthsum.
        '''      
        cardinality_birth = np.zeros(self.model['n_max']+1) # Initiate a list with length n_max, the maximum number of targets per frame.
        intensity_lambda =self.model['w_birthinit']/self.model['n_birth']

        for n in range(self.model['n_max']+1):
            # PPP RFS distribution's set integral is Poisson Distribution, which is why the cardinality distribution of PPP(new birth targets) is a poisson distribution.
            # Thus here we calculate cardinality distribution of new birth targets by using pmf of poission distribution.
            # To refresh poisson distribution pmf, refer to: https://en.wikipedia.org/wiki/Poisson_distribution.
            cardinality_birth[n]=(intensity_lambda**n) * np.exp(-intensity_lambda)/math.factorial(n)

        cardinality_pred = cardinality_birth

        return cardinality_pred

    def intensity_predict(self, prunedIntensity):
        '''
        Eq. 14 of [2] compute the predicted intensity
        This step is the same as that of GM-PHD, with no cardinality parameters
        '''
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        v_init = [0.0, 0.0] # initial velocity
        w_birthsum = self.model['w_birthsum']
        n_birth = self.model['n_birth']

        # Prediction for new birth targets. See step 1 in table I in [1]. But beware our current implementation is NOT exact same
        # as how it is implemented in step 1 of table I in [1].
        # For fixed numebr of new birth targets born from the fixed places every frame assumption.
        for birth_index in range(n_birth):
            position = [100 + birth_index * 10, 100 + birth_index * 10]
            w.append(w_birthsum/n_birth)
            m.append(np.array([position[0], position[1], v_init[0], v_init[1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.   
        
        # Prediction for surviving(existing) targets. See step 2 in table I in [1].
        numTargets_Jk_minus_1 = len(prunedIntensity['w']) # Number of Gaussian components after the pruning and merging step at last frame(stands for number of surviving/existing targets).
        for i in range(numTargets_Jk_minus_1): # For each Gaussian components(existing/surviving targets from last frame)
            # Calculate the predicted weights for surviving targets.
            w.append(self.model['p_S']*prunedIntensity['w'][i]) # Equation (25) in [1].
            
            # Calculate the predicted mean, m, and predicted covariance, P, for surviving targets by performing Kalman filter.
            if self.motion_model_type == "Constant Velocity":    
                m.append(self.model['F_k'].dot(prunedIntensity['m'][i]).astype('float64'))  # Equation (26) in [1].
                P.append(self.model['Q_k'] + self.model['F_k'].dot(prunedIntensity['P'][i]).dot(np.transpose(self.model['F_k'])))   # Equation (27) in [1].
            elif self.bayesian_filter_type == "Constant Accelaration": # If we have other type of motion model.
                pass
            elif self.bayesian_filter_type == "Constant Turning":
                pass
            elif self.bayesian_filter_type == "IMM":
                pass
            else:
                IOError("The configed motion model Not available!")
            # TODO: Now the prediction step is just multiplied by using 'F_k' which is based on constant velocity(CV) motion 
                # model as we set up the model in function gen_model. However, we should consider handling more state transition model
                # and even using mixed motion model to come up with better prediction step result for NuScenes dataset which recorded 
                # in real world scenario.
                # -- Another additional issue is to use Interacting Multiple Motion filter(IMM) which employs several different 
                # motion models(CP, CV, CA, CT) by using several sub filter(the sub filter could be EKF). The output of IMM is 
                # fusion of state estimations of all sub filters(Sort of combination of motion hypotheses).
                # -- Another possible issue is, should we also think about if we need better nonlinear expression of motion model?
                # motion(state transition matrix) model should be linearized, if we have nonlinear motion model for setting up the 
                # model in function gen_model.

        predictedIntensity = {}
        predictedIntensity['w'] = w
        predictedIntensity['m'] = m
        predictedIntensity['P'] = P

        return predictedIntensity

    def intensity_predict_initial_step(self):
        '''
        Compute the predicted intensity for the initial step.
        This one only has the birth part of predicted intensity, without the survival intensity part(Due to no surviving targets at initial step).
        Also, the intensity used here is w_birthinit instead of w_birthsum.
        '''
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        v_init = [0.0, 0.0] # initial velocity
        w_birthinit = self.model['w_birthinit']
        
        # For fixed numebr of new birth targets born from the fixed places every frame assumption.
        for birth_index in range(self.model['n_birth']):
            position = [100 + birth_index * 10, 100 + birth_index * 10]
            w.append(w_birthinit/self.model['n_birth'])
            m.append(np.array([position[0], position[1], v_init[0], v_init[1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.   
      
        predictedIntensity = {}
        predictedIntensity['w'] = w
        predictedIntensity['m'] = m
        predictedIntensity['P'] = P

        return predictedIntensity

    """
    STEP 2: Construct components for both gating and intensity update.
    """
    def construct_components_for_gating_and_intensity_update_step(self, predictedIntensity):
        # Construct components for PHD update step. This step is the same as that of PHD update. (step 3 of table I in original GM-PHD filter paper[1])
        eta = []
        S = []
        K = []
        P = []
        for i in range(len(predictedIntensity['w'])):
            eta.append(self.model['H_k'].dot(predictedIntensity['m'][i]).astype('float64')) # Calculate predicted measurements.
            S.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['P'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S[i])
            if self.model['using_cholsky_decomposition_for_calculating_inverse_of_measurement_covariance_matrix'] == True:
                Vs = np.linalg.cholesky(np.array(Si, dtype=np.float64))
                inv_sqrt_S = np.linalg.inv(Vs)
                invSi = inv_sqrt_S.dot(np.transpose(inv_sqrt_S))
            else:
            
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function

            K.append(predictedIntensity['P'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P.append(predictedIntensity['P'][i] - K[i].dot(self.model['H_k']).dot(predictedIntensity['P'][i]).astype('float64'))


        constructUpdateIntensity = {}
        constructUpdateIntensity['eta'] = eta
        constructUpdateIntensity['S'] = S
        constructUpdateIntensity['K'] = K
        constructUpdateIntensity['P'] = P        
        
        return constructUpdateIntensity

    """
    OPTIONAL STEP: Gating
    """
    # Only take the measurements/detections within gate into account, depends on sort of "distance" between measuremtns 
    # and predicted states of targets.
    def gateMeasurements(self, Z_k, constructUpdateIntensity, use_gating = True):
        # We borrow the definition of validation of measurement from the paper.
        # 2010 The probabilistic data association filter, IEEE control systems.
        # the validation step follows Equation (31) in [1].
        # the elimination step follows equation (34) in [1].
        threshold = self.model['gating_threshold']
        S = constructUpdateIntensity['S']
        # notice the physical meaning of eta, HFX which is the noiseless connection between the states and measurements
        eta = constructUpdateIntensity['eta']
        
        if use_gating == True:
            Z_k_gated = []
            Z_k_gated_position = []
            for j in range(len(eta)): # loop over all existing targets
                for z in range(len(Z_k)): # loop over all measurements
                    position_difference = eta[j][:2] - Z_k[z][:2] 
                    z_position = (Z_k[z][0],Z_k[z][1])
                    # if the measurement falls into the valid region of a track and it has not been included in the list
                    if np.transpose(position_difference)@np.linalg.inv(S[j])@(position_difference) < threshold and z_position not in Z_k_gated_position:
                        Z_k_gated.append(Z_k[z])
                        Z_k_gated_position.append(z_position)
        else:
            Z_k_gated = Z_k
        return Z_k_gated

    """
    STEP 3: Compute the upsilon
    """
    
    def compute_upsilon(self,predictedIntensity, constructUpdateIntensity,Z_k):
        N_measurements = len(Z_k)
        measurement_likelihood_matrix = compute_measurement_likelihood_matrix(constructUpdateIntensity, Z_k)        
        '''
        Elementary Symmetric Functions as described by section IV B of [2].
        '''
        # Input of the elementary symmetric functions.
        Lambda_upd=np.zeros(N_measurements)
        for i in range(N_measurements):
            # unitclutter intensity is used here, which is 1/area
            # for more information, please refer to Garsia's code GMCPHD_update.m line 81
            # and Vo's code in cphd folder run_filter.m line 120
            Lambda_upd[i]=self.model['p_D']*np.inner(predictedIntensity['w'],measurement_likelihood_matrix[:,i])/self.model['unitclutterIntensity']
        '''
        Garcia's Matlab original code
        Lambda_upd=zeros(N_measurements,1);
        for i=1:N_measurements    
        Lambda_upd(i)=p_d*weights_pred'*weight_z(:,i)*(Area(1)*Area(2)); weights_pred&weight_z(:,i) are both column vector
        end
        '''
        # Calculate the elementary symmetric functions for all elements of Lambda_upd and with one element removed.
        esf_all=compute_elementary_symmetric_functions(Lambda_upd)
        esf_all_minus_one=np.zeros((N_measurements,N_measurements))
        for i in range(N_measurements):
            Lambda_upd_first = Lambda_upd[:i]
            Lambda_upd_second = Lambda_upd[i+1:]
            Lambda_upd_i= np.hstack((Lambda_upd_first,Lambda_upd_second)) # tease out i
            esf_all_minus_one[:,i]=compute_elementary_symmetric_functions(Lambda_upd_i)
        '''
        Garcia's Matlab original code
        %We calculate the elementary symmetric functions for all elements
        %of Lambda_upd and with one element removed
        esf_all=Compute_esf(Lambda_upd);
        esf_all_minus_one=zeros(N_measurements,N_measurements);
        for i=1:N_measurements    
            Lambda_upd_i=[Lambda_upd(1:i-1);Lambda_upd(i+1:end)];
           esf_all_minus_one(:,i)=Compute_esf(Lambda_upd_i);
        end
        '''

        '''
        Compute Upsilon with symmetric functions.
        equation (21) of [2]
        '''
        # Initiate upsilons 
        upsilon0=np.zeros(self.model['n_max']+1)
        upsilon1=np.zeros(self.model['n_max']+1)
        upsilon1_all_minus_one=np.zeros((self.model['n_max']+1,N_measurements))

        for n in range(self.model['n_max']+1):
            # Compute Upsilon0
            #l_clutter = self.model['lambda_t'] # Number of clutters in area follows Poisson distribution
            l_clutter=self.model['average_number_of_clutter_per_frame']
            for j in range(min(N_measurements,n)+1): # if there are more measurements than n_max only take n_max measurements
                upsilon0[n]+=np.exp(-l_clutter
                                    +(N_measurements-j)*np.log(l_clutter)
                                    +sum(np.log(x+1) for x in range(n)[n-j:])
                                    +(n-j)*np.log(1-self.model['p_D'])
                                    -j*np.log(sum(predictedIntensity['w']))
                                   )*esf_all[j]
            '''
            Garcia's Matlab original code
            upsilon0(n+1)=upsilon0(n+1)
            +exp(-l_clutter
                +(N_measurements-j)*log(l_clutter)
                +sum(log(n-j+1:n))
                +(n-j)*log(1-p_d)
                -j*log(sum(weights_pred))
                )*esf_all(j+1);
            '''
            # Compute Upsilon1
            for j in range(min(N_measurements,n-1)+1):
                upsilon1[n]+=np.exp(-l_clutter
                                    +(N_measurements-j)*np.log(l_clutter)
                                    +sum(np.log(x+1) for x in range(n)[n-j:])
                                    +(n-(j+1))*np.log(1-self.model['p_D'])
                                    -(j+1)*np.log(sum(predictedIntensity['w']))
                                   )*esf_all[j]
            '''
            Garcia's Matlab original code        
            upsilon1(n+1)=upsilon1(n+1)+exp(
                -l_clutter
                +(N_measurements-j)*log(l_clutter)
                +sum(log(n-j:n))
                +(n-(j+1))*log(1-p_d)-(j+1)*log(sum(weights_pred))
                )*esf_all(j+1);
            '''
            # upsilon1_all_minus_one
            for i in range(N_measurements):
                for j in range(min((N_measurements-1),n-1)+1):
                    upsilon1_all_minus_one[n,i]+=np.exp(-l_clutter
                                                        +((N_measurements-1)-j)*np.log(l_clutter)
                                                        +sum(np.log(x) for x in range(n+1)[n-j:])
                                                        +(n-(j+1))*np.log(1-self.model['p_D'])
                                                        -(j+1)*np.log(sum(predictedIntensity['w']))
                                                       )*esf_all_minus_one[j][i]    

            '''
            Garcia's Matlab original code
            upsilon1_all_minus_one(n+1,i)=upsilon1_all_minus_one(n+1,i)+
            exp(-l_clutter+
               ((N_measurements-1)-j)*log(l_clutter)
               +sum(log(n-j:n))
               +(n-(j+1))*log(1-p_d)
               -(j+1)*log(sum(weights_pred))
               )*esf_all_minus_one(j+1,i);          
            '''
       
        return upsilon0, upsilon1, upsilon1_all_minus_one
    '''
    Garcia's Matlab  code for upsilon computation
    %We calculate all upsilons % EQUATION 21
    upsilon0=zeros(Ncardinality_max+1,1);
    upsilon1=zeros(Ncardinality_max+1,1);
    upsilon1_all_minus_one=zeros(Ncardinality_max+1,N_measurements);
    for n=0:Ncardinality_max
        %Calculate upsilon0
        for j=0:min(N_measurements,n)    
        upsilon0(n+1)=upsilon0(n+1)+exp(-l_clutter+(N_measurements-j)*log(l_clutter)+sum(log(n-j+1:n))+(n-j)*log(1-p_d)-j*log(sum(weights_pred)))*esf_all(j+1);
        end
        %Calculate upsilon1
        for j=0:min(N_measurements,n-1)    
        upsilon1(n+1)=upsilon1(n+1)+exp(-l_clutter+(N_measurements-j)*log(l_clutter)+sum(log(n-j:n))+(n-(j+1))*log(1-p_d)-(j+1)*log(sum(weights_pred)))*esf_all(j+1);
        end
        %Calculate upsilon1_all_minus_one
        for i=1:N_measurements
            for j=0:min((N_measurements-1),n-1)
                upsilon1_all_minus_one(n+1,i)=upsilon1_all_minus_one(n+1,i)+exp(-l_clutter+((N_measurements-1)-j)*log(l_clutter)+sum(log(n-j:n))+(n-(j+1))*log(1-p_d)-(j+1)*...
                    log(sum(weights_pred)))*esf_all_minus_one(j+1,i);          
            end
        end
    end
    '''

    """
    STEP 4: Update
    """   
    def cardinality_update(self,upsilon0, cardinality_pred):
        '''
        equation (19) in [2]
        Input:
        cardinality_pred: a list with length n_max 
        Return:
        updated_cardinality: a list with length n_max
        '''
        updated_cardinality = []
        for i in range(self.model['n_max']+1):
            updated_cardinality.append(upsilon0[i]*cardinality_pred[i]) 
        updated_cardinality=updated_cardinality/sum(updated_cardinality)
        # The alternative way to calculate updated_cardinality is as below:
        """ updated_cardinality = [x*y for x, y in zip(upsilon0,cardinality_pred)]/np.inner(upsilon0, cardinality_pred) """
        '''
        Garcia's Matlab original code
        %Cardinality update 
        cardinality_u=upsilon0.*cardinality_pred; %EQUATION 19
        cardinality_u=cardinality_u/sum(cardinality_u); %EQUATION 19
        '''
        return updated_cardinality

    def intensity_update(self, Z_k, predictedIntensity,cardinality_pred, upsilon0, upsilon1, upsilon1_all_minus_one,constructUpdateIntensity,):
        """
        equation (20) in [2]. It is a summation of two parts:
        1) miss-detected targets(undetected targets, whose actual RFS is a Poission point process), modified by a proportional term derived from upsilons.
        2) detected targets(whose actual RFS is a multi-Bernoulli RFS), modified by a proportional term derived from upsilons. 
            A) Construction of all elements belong to single Bernoulli component, and 
            B) Based on all elements belong to single Bernoulli component, we calculate all parameters of 
                update step of PHD for the every single Bernoulli RFS component.
        """
        # "Miss-detected targets" part of GM-CPHD update. This is the first term of Eq. 20 of [2].
        # We scale all weights by probability of missed detection (1 - p_D)
        w = []
        m = []
        P = []
        coupling_coefficient_for_miss_detected_target = np.inner(upsilon1, cardinality_pred)/np.inner(upsilon0,cardinality_pred)
        for i in range(len(predictedIntensity['w'])):
            # Calculate the updated weights for missed-detected targets(Only rely on the predicted result due that no measurements detected),
            # which is scaled by the "coupling coefficient for miss detected target" compare to PHD update step, w.append((1.0 - self.model['p_D'])*predictedIntensity['w'][i])
            w.append((1.0 - self.model['p_D'])*predictedIntensity['w'][i]*coupling_coefficient_for_miss_detected_target) # The first part of equation (20) in [2], which is for non-detected targets.
            '''
            Garcia's Matlab original code
            weights_u(1:Ncom_k)=(upsilon1'*cardinality_pred)/(upsilon0'*cardinality_pred)*(1-p_d)*weights_pred;
            '''
            # Calculate the updated mean, m, and covariance, P, for missed-detected targets(Only rely on the predicted result due that no measurements detected)
            m.append(predictedIntensity['m'][i])
            P.append(predictedIntensity['P'][i])

        # "Detected targets" part of GM-CPHD update. Notice this part is just mplementation of second "for loop" in step 4 in 
        # table I in original GM-PHD filter paper[1](w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, 
        # constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j]))) but scaled by the "coupling coefficient for 
        # detected target".
        # Every observation updates every target(to form all possible hypotheses for data association).
        numTargets_Jk_k_minus_1 = len(predictedIntensity['w']) # Number of Gaussian components after the prediction step
        l = 0
        for z in range(len(Z_k)): # Iterate over m measurements/detection points
            coupling_coefficient_for_detected_target = np.inner(upsilon1_all_minus_one[:,z],cardinality_pred)/np.inner(upsilon0,cardinality_pred)
            l = l + 1   # l actually stands for l^th measurement. Thus l is always equal to z + 1.
            for j in range(numTargets_Jk_k_minus_1):
                z_k = copy.deepcopy(Z_k[z])
                # The second part of equation (20) in [2], which is for detected targets
                # unitclutter intensity is used here, which is 1/area
                # for more information, please refer to Garsia's code GMCPHD_update.m line 123
                # and Vo's code in cphd folder run_filter.m line 173
                w.append(self.model['p_D']*predictedIntensity['w'][j]*multivariate_normal.pdf([z_k[0][0],z_k[1][0]], [constructUpdateIntensity['eta'][j][0][0],constructUpdateIntensity['eta'][j][1][0]], constructUpdateIntensity['S'][j])*coupling_coefficient_for_detected_target/self.model['unitclutterIntensity'])
                # Alternatively, we could also use mvnpdf function to replace the multivariate_normal.pdf function provided by scipy library API 
                """ w.append(self.model['p_D']*predictedIntensity['w'][j]*mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j])*coupling_coefficient_for_detected_target) """ # The second part of equation (20) in [2], which is for detected targets
                '''
                Garcia's Matlab original code
                for i=1:N_measurements
                    weights_u(indeces_u_detect(:,i))=(upsilon1_all_minus_one(:,i)'*cardinality_pred)/(upsilon0'*cardinality_pred)
                    *p_d.*weight_z(:,i)*(Area(1)*Area(2)).*weights_pred;
                end
                '''
                if self.bayesian_filter_type == "Kalman":
                    # Calculate the updated mean, m, and updated covariance, P, for surviving targets by performing Kalman filter.
                    m.append(predictedIntensity['m'][j] + constructUpdateIntensity['K'][j].dot(z_k - constructUpdateIntensity['eta'][j]).astype('float64'))
                    P.append(constructUpdateIntensity['P'][j])
                elif self.bayesian_filter_type == "EKF": # If we have nonlinear measurement model, we should use EKF update
                    # TODO: Now the update step is just multiplied by using 'H_k' which is based on observation model when we set 
                    # up the model in function gen_model. We should consider handling nonlinear observation model to come up
                    # with better update step result for NuScenes dataset which recorded in real world scenario, if the measurement 
                    # model is changed to nonlinear measurement model.  
                    # -- One possible solution is to use EKF instead of linear Kalman filter, handling better nonlinear expression of 
                    # observation model. (However, since in our plan, the detections/measurements should be in format of 'x, y, l, w, theta'
                    # which represents x, y position of center point, length and width, and orientation directly, it seems our measurement
                    # model will always be linear measurement model H = np.eye() rather than nonlinear measurement model in our task)
                    pass
                elif self.bayesian_filter_type == "Particle filter":
                    pass
                else:
                    IOError("The configed Baysian filter Not available!")

            #total_w_d = 0.0
            #for j in range(numTargets_Jk_k_minus_1):
            #    total_w_d += w[l*numTargets_Jk_k_minus_1 + j]

            #for j in range(numTargets_Jk_k_minus_1):
            #    k_k = self.model['clutterIntensity']
            #    w[l*numTargets_Jk_k_minus_1 + j] = w[l*numTargets_Jk_k_minus_1 + j] / (k_k + total_w_d) # Updated weight by normalization
 
        # Combine both miss-detected targets and detected targets part of the GM-PHD update.
        updatedIntensity = {}
        updatedIntensity['w'] = w
        updatedIntensity['m'] = m
        updatedIntensity['P'] = P

        return updatedIntensity

    # Beware that until now, the weight vector w_update, which represents all the weights for all Gaussian components in the Gaussian mixture
    # which represents PHD(intensity) of posterior Poisson approximation of Multi-Bernoulii RFS, has "number_of_measurements*(number_of_existing_tracks + 1)"
    # components(and corresponding weights values), each of components represents one possible data association hypothesis.

    """
    STEP 5: Pruning, Merging.
    See section IV C for detailed description. This step is the same for both GM-PHD and GM-CPHD.
    See table II in [1] for detailed implementation steps.
    """
    def pruneAndMerge(self, updatedIntensity):
        w = []
        m = []
        P = []

        # Prune out the low-weighted components, which is simple apporach to prune/truncate Gaussian components in the Gaussian miture. 
        # Only keep all the Gaussian components whose weights are larger than elim_threshold.
        I = [index for index,value in enumerate(updatedIntensity['w']) if value > self.model['T']]  # Indices of large enough weights
        # I = np.where(np.array(updatedIntensity['w']) >= self.model['T'])[0]

        # Merge the close-together components, which is simple apporach to merge, Gaussian components in the Gaussian miture which are 
        # close to each others, together. However, there will be chance to merge two Gaussian components belong to two targets which 
        # are fairly close with each other, as one Gaussian component.
        while len(I) > 0:
            highWeights = np.array(updatedIntensity['w'])[I]
            j = np.argmax(highWeights)
            j = I[j]
            # Find all points with Mahalanobis distance less than merge distance threshold, U, from point updatedIntensity['m'][j]
            L = []  # A vector of indices of merged Gaussians.
            for iterI in range(len(I)):
                thisI = copy.deepcopy(I[iterI])
                delta_m = updatedIntensity['m'][thisI] - updatedIntensity['m'][j]
                mahal_dist = np.transpose(delta_m).dot(np.linalg.inv(np.array(updatedIntensity['P'][thisI],dtype=np.float64))).dot(delta_m)
                if mahal_dist <= self.model['U']:
                    L.append(thisI)  # Indices of merged Gaussians

            # The new weight of the resulted merged Guassian is the summation of the weights of the Gaussian components.
            w_bar = sum(np.array(updatedIntensity['w'])[L])
            w.append(w_bar)

            # The new mean of the merged Gaussian is the weighted average of the merged means of Gaussian components.
            m_val = []
            for i in range(len(L)):
                thisI = copy.deepcopy(L[i])
                m_val.append(updatedIntensity['w'][thisI]*updatedIntensity['m'][thisI])
            m_bar = sum(m_val)/w_bar
            m.append(m_bar.astype('float64'))

            # Calculating covariance P_bar is a bit trickier
            P_val = []
            for i in range(len(L)):
                thisI = copy.deepcopy(L[i])
                delta_m = m_bar - updatedIntensity['m'][thisI]
                P_val.append(updatedIntensity['w'][thisI]*(updatedIntensity['P'][thisI] + delta_m.dot(np.transpose(delta_m))))
            P_bar = sum(P_val)/w_bar
            P.append(P_bar.astype('float64'))

            # Now delete the elements in L from I
            for i in L:
                I.remove(i)
        # capping guassian components if there are more gaussians than the maximum number allowed.
        prunedMergedIntensity = {}
        if self.model['capping_gaussian_components'] == True:
            if len(w)>self.model['maximum_number_of_gaussian_components']:
                w_cap = []
                m_cap = []
                P_cap = []
                index_of_ranked_weights_in_ascending_order=np.argsort(w)
                index_of_ranked_weights_in_descending_order = np.flip(index_of_ranked_weights_in_ascending_order)

                for i in range(len(index_of_ranked_weights_in_descending_order)):
                    idx = index_of_ranked_weights_in_descending_order[i]
                    w_cap.append(w[idx])
                    m_cap.append(m[idx])
                    P_cap.append(P[idx])
     
                prunedMergedIntensity['w'] = w_cap
                prunedMergedIntensity['m'] = m_cap
                prunedMergedIntensity['P'] = P_cap
            else:
                prunedMergedIntensity['w'] = w
                prunedMergedIntensity['m'] = m
                prunedMergedIntensity['P'] = P         

        else:
            prunedMergedIntensity['w'] = w
            prunedMergedIntensity['m'] = m
            prunedMergedIntensity['P'] = P

        return prunedMergedIntensity

    """
    STEP 6: Extracting estimated states (Multi targets state extraction)
    See section III C for detailed description.
    Notice that a key difference between GM-CPHD and GM-PHD is, cardinality is unknown in GM-PHD. A
    threshold of w_h is established in order to tease out distributions contributes little to the overall distribution.
    Yet for GM-CPHD, cardinality is derived from the cardinality distribution. With a known cardinality,
    the state extraction step is simply sorting distributions with N largest weight
    There are more than one way to implement the sorting of a list and extracting the largest N elements,.
    Here we choose to implement the described process(section III C) verbatim. 
    """
    def extractStates(self, PrunedAndMerged, updated_cardinality):
        '''
        Garcia's Matlab original code
        [max_card,index_card]=max(cardinality_u);
        N_detections=index_card-1;
        if(N_detections>0)
        X_estimate=zeros(size(means_u,1),N_detections);    
        for i=1:N_detections
            [max_weight,index_max]=max(weights_u);
            X_estimate(:,i)=means_u(:,index_max);
            weights_u(index_max)=0;   
        end
        X_estimate=X_estimate(:);
        else
        X_estimate=[];
        end
        '''
        w = [] # Initiate an empty list of weights of gaussian distributions.
        m = [] # Initiate an empty list of mean of gaussian distributions.
        P = [] # Initiate an empty list of covariance pf gaussian distributions.

        """w_PrunedAndMerged = copy.deepcopy(PrunedAndMerged['w'])"""
        max_cardinality_index = np.argmax(updated_cardinality)
        negative_weight = [-PrunedAndMerged['w'][x] for x in range(len(PrunedAndMerged['w']))]
        descending_order_index = np.argsort(negative_weight)
        
        cardinality = 0
        for idx in descending_order_index:
            if cardinality < max_cardinality_index:
                #max_weight = max(PrunedAndMerged['w']) # get the distribution with the max weight
                w.append(PrunedAndMerged['w'][idx]) # append the max weight to w
                m.append(PrunedAndMerged['m'][idx]) # append the mean of this distribution to m
                P.append(PrunedAndMerged['P'][idx]) # append the P of this distribution to P
                cardinality+=1
        
        extractedStates = {}
        extractedStates['w'] = w # N_tracks guassian distributions, weights
        extractedStates['m'] = m # N_tracks gaussian distributions, mean
        extractedStates['P'] = P # N_tracks gaussian distributions, covariance

        return extractedStates
