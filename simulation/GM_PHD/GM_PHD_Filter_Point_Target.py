"""
%% ------------------------------ Gaussian Mixture(GM) Probability Hypothesis Density(PHD) filter ------------------------------ %%
This Python code is reproduction for the "point target GM-PHD filter" originally proposed in paper [1], with assumption
of no target spawning. The original Matlab code for "point target GM-PHD filter" could be available from authors website
http://ba-tuong.vo-au.com/codes.html

%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
Processing
"""
import numpy as np
import copy
from scipy.stats import multivariate_normal
import math
from util import mvnpdf

"""
GM-PHD Point Target Filter
"""
class GM_PHD_Filter:
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
    def predict(self, Z_k, prunedIntensity):
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        v_init = [0.0, 0.0] # initial velocity
        w_birthsum = self.model['w_birthsum']
        z_init = self.model['z_init'][0]
        n_birth = self.model['n_birth']

        # Prediction for new birth targets. See step 1 in table I in [1]. But beware our current implementation is NOT exact same
        # as how it is implemented in step 1 of table I in [1].
        # TODO: For now we assume all the measurements might be origined from 'new birth targets', and calculate the weight, mean 
        # and covariance for each of them. Obviously it is not a good idea, we have to figure out a good way to estimate how many 
        # the 'new birth targets' and where they possible are located roughly for observation.
        # One possible way to better handle the 'new birth target', is, we assume all the measurements which have not been associated
        # to any surviving targets are new birth targets(same as how we usually handle new birth target in pure Bayesian filter based
        # tracker). However, such approach may NOT be possible since we do NOT really have explicit data association in PHD filter.
        # Another possible way is like how this step is implemented in step 1 of table I in [1], just hardcode value for "assumed 
        # number of new birth targets and assumed corresponding birth locations represented in mean and variance", and "number of 
        # spawning targets".
        for i in range(n_birth):
            w.append(w_birthsum)
            m.append(np.array([z_init[0], z_init[1], v_init[0], v_init[1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
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
        predictedIntensity['w'] = w
        predictedIntensity['m'] = m
        predictedIntensity['P'] = P

        return predictedIntensity

    """
    Step 3 and 4 of table I in [1]: Construct components for PHD update step and update both targets without measurements(missing detected targets) 
    and targets with measurements(detected targets).
        -- For miss-detected targets(undetected targets, whose actual RFS is a Poission point process).
        -- For detected targets(whose actual RFS is a multi-Bernoulli RFS).
            A) Construction of all elements belong to single Bernoulli component, and 
            B) Based on all elements belong to single Bernoulli component, we calculate all parameters of 
                update step of PHD for the every single Bernoulli RFS component.
        Notice this part of code is to implementation of step 3 and "calculation parameters of update step of PHD, 
        i.e., m_k and P_k for every single Bernoulli RFS component" of step 4 in table I in original GM-PHD filter paper[1].
    """
    def update(self, Z_k, predictedIntensity):
        # Construct components for PHD update step. (step 3 of table I in original GM-PHD filter paper[1])
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

        # flag will be used to save the index of every target(both miss-detected targets and detected targets.), to distinguish every target. This
        # information will be used later in merge part to only allow Gaussian components belong to the same target to be merged together, if we 
        # configure the "self.model['only_merge_the_Gaussian_components_belong_to_same_target_together']" == True.
        flag=[]

        # "Miss-detected targets" part of GM-PHD update. Notice this part is just mplementation of first "for loop" in step 4 in 
        # table I in original GM-PHD filter paper[1].
        # We scale all weights by probability of missed detection (1 - p_D)
        w = []
        m = []
        P = []
        for i in range(len(predictedIntensity['w'])):
            # Calculate the updated weights for missed-detected targets(Only rely on the predicted result due that no measurements detected)
            w.append((1.0 - self.model['p_D'])*predictedIntensity['w'][i])
            # Calculate the updated mean, m, and covariance, P, for missed-detected targets(Only rely on the predicted result due that no measurements detected)
            m.append(predictedIntensity['m'][i])
            P.append(predictedIntensity['P'][i])
            flag.append(i)

        # "Detected targets" part of GM-PHD update. Notice this part is just mplementation of second "for loop" in step 4 in 
        # table I in original GM-PHD filter paper[1].
        # Every observation updates every target(to form all possible hypotheses for data association).
        numTargets_Jk_k_minus_1 = len(predictedIntensity['w']) # Number of Gaussian components after the prediction step
        l = 0
        clutterIntensity = self.model['clutterIntensity']
        threshold = self.model['gating_threshold']
     
        for z in range(len(Z_k)): # Iterate over m measurements/detection points
            l = l + 1   # l actually stands for l^th measurement. Thus l is always equal to z + 1.
            for j in range(numTargets_Jk_k_minus_1):
                flag.append(j)
                z_k = copy.deepcopy(Z_k[z])
                if self.model['use_gating'] == True: # Apply gating.
                    position_difference = eta[j][:2] - z_k[:2] 
                    # if the current measurement falls into the valid region of current detected track and it has not been included in the list
                    if np.transpose(position_difference)@np.linalg.inv(S[j])@(position_difference) < threshold:
                        # w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k[0:2], constructUpdateIntensity['eta'][j][0:2], constructUpdateIntensity['S'][j][0:2, 0:2]))  # Hoping multivariate_normal.pdf is the right one to use; this is for 2D bounding box [x, y, w, h]
                        w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j]))  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                        #w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j])/clutterIntensity)  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]

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
                    #else: # If current measurement does not fall in gating area of current detected target, do not in update, only remain same as the predicted intensity.
                        # But we set the weight(i.e. the existing probability of this Gaussian component) to be a very small value.
                        #w.append(2e-12)
                        #""" w.append(predictedIntensity['w'][j]) """  # Should we set the weight same as the predicted weight for this Gaussian component?
                        #""" w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j])) """  # Should we still update the weight normally?
                        #m.append(predictedIntensity['m'][i])
                        #P.append(predictedIntensity['P'][i])
                else: 
                    w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j])) 
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
            
            total_w_d = 0.0
            for j in range(numTargets_Jk_k_minus_1):
                total_w_d += w[l*numTargets_Jk_k_minus_1 + j]

            for j in range(numTargets_Jk_k_minus_1):
                w[l*numTargets_Jk_k_minus_1 + j] = w[l*numTargets_Jk_k_minus_1 + j] / (clutterIntensity + total_w_d) # Updated weight by normalization
                #w[l*numTargets_Jk_k_minus_1 + j] = w[l*numTargets_Jk_k_minus_1 + j] / total_w_d # Updated weight by normalization

        # Combine both miss-detected targets and detected targets part of the GM-PHD update.
        updatedIntensity = {}
        updatedIntensity['w'] = w
        updatedIntensity['m'] = m
        updatedIntensity['P'] = P
        updatedIntensity['flag'] = flag

        return updatedIntensity

    # Beware that until now, the weight vector w_update, which represents all the weights for all Gaussian components in the Gaussian mixture
    # which represents PHD(intensity) of posterior Poisson approximation of Multi-Bernoulii RFS, has "number_of_measurements*(number_of_existing_tracks + 1)"
    # components(and corresponding weights values), each of components represents one possible data association hypothesis.

    """
    Step 5: Pruning, Merging.
    See table II in [1] for more info.
    """
    def pruneAndMerge(self, updatedIntensity):
        w = []
        m = []
        P = []

        # Prune out the low-weighted components, which is simple apporach to prune/truncate Gaussian components in the Gaussian miture. 
        # Only keep all the Gaussian components whose weights are larger than elim_threshold.
        I = [index for index,value in enumerate(updatedIntensity['w']) if value > self.model['T']]  # Indices of large enough weights
        # I = np.where(np.array(updatedIntensity['w']) >= self.model['T'])[0]

        # Merge the close-together components, which is simple apporach to merge, Gaussian components in the Gaussian mixture which are 
        # close to each others, together. However, there will be chance to merge two Gaussian components belong to two targets which 
        # are fairly close with each other, as one Gaussian component.
        while len(I) > 0:
            highWeights = np.array(updatedIntensity['w'])[I]
            index_of_this_component_in_pruned_list = np.argmax(highWeights)
            index_of_this_component_in_original_list = I[index_of_this_component_in_pruned_list]
            # Find all points with Mahalanobis distance less than merge distance threshold, U, from point updatedIntensity['m'][j]
            L = []  # A vector of indices of merged Gaussians.
            for iterI in range(len(I)):
                thisI = copy.deepcopy(I[iterI])
                delta_m = updatedIntensity['m'][thisI] - updatedIntensity['m'][index_of_this_component_in_original_list]
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

                for i in range(self.model['maximum_number_of_gaussian_components']):
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
    Step 6: Extracting estimated states (Multi targets state extraction)
    See table III in [1] for more info.
    """
    def extractStates(self, PrunedAndMerged):
        w = []
        m = []
        P = []
        for i in range(len(PrunedAndMerged['w'])):
            if PrunedAndMerged['w'][i] > self.model['w_thresh']:
                for j in range(int(round(PrunedAndMerged['w'][i]))):  # If a target has a rounded weight greater than 1, output it multiple times.
                   w.append(PrunedAndMerged['w'][i])
                   m.append(PrunedAndMerged['m'][i])
                   P.append(PrunedAndMerged['P'][i])
        '''
        # Alternatively, this step can be done in accordance with
        # Estimator based on Section 9.5.4.4 in R. P. S. Mahler, Advances in Statistical 
        # Multisource-Multitarget Information Fusion. Artech House, 2014
        w_sum = 0
        for i in range(len(PrunedAndMerged)):
            w_sum += PrunedAndMerged['w'][i]
        cardinality = round(w_sum)
        print('cardinality of this frame is {}'.format(cardinality))

        if cardinality > 0:
            for n in range(cardinality):
                max_weight_index = np.argmax(PrunedAndMerged['w']) # get the index of the distribution with the max weight
                max_weight = PrunedAndMerged['w'][max_weight_index] # get the distribution with the max weight
                w.append(max_weight) # append the max weight to w
                m.append(PrunedAndMerged['m'][max_weight_index]) # append the mean of this distribution to m
                P.append(PrunedAndMerged['P'][max_weight_index]) # append the P of this distribution to P
                PrunedAndMerged['w'][max_weight_index] = -1 # set the max weight to be 0 such that this set of data is marked as used     
        '''
        extractedStates = {}
        extractedStates['w'] = w
        extractedStates['m'] = m
        extractedStates['P'] = P

        return extractedStates
