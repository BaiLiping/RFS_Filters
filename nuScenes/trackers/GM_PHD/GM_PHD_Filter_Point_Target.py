"""
%% ------------------------------ Gaussian Mixture(GM) Probability Hypothesis Density(PHD) filter ------------------------------ %%
This Python code is reproduction for the "point target GM-PHD filter" originally proposed in paper [1], with assumption
of no target spawning. The original Matlab code for "point target GM-PHD filter" could be available from authors website
http://ba-tuong.vo-au.com/codes.html

%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
Processing
"""
from json import detect_encoding
import numpy as np
import copy
from scipy.stats import multivariate_normal
import math
from trackers.GM_PHD.util import mvnpdf

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
    def predict(self,Z_k,  prunedIntensity, birth_rate):
        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        elevation=[]
        size=[]
        rotation=[]
        velocity=[]
        detection_score=[]
        classification=[]
        id=[]
        max_id=prunedIntensity['max_id']
        # Prediction for surviving(existing) targets. See step 2 in table I in [1].
        numTargets_Jk_minus_1 = len(prunedIntensity['w']) # Number of Gaussian components after the pruning and merging step at last frame(stands for number of surviving/existing targets).
        for i in range(numTargets_Jk_minus_1): # For each Gaussian components(existing/surviving targets from last frame)
            # Calculate the predicted weights for surviving targets.
            w.append(self.model['p_S']*prunedIntensity['w'][i]) # Equation (25) in [1].
            #w.append(prunedIntensity['detection_score'][i]*prunedIntensity['w'][i]) # Equation (25) in [1].
            # Calculate the predicted mean, m, and predicted covariance, P, for surviving targets by performing Kalman filter.
            m.append(self.model['F_k'].dot(prunedIntensity['mean'][i]).astype('float64'))  # Equation (26) in [1].
            elevation.append(prunedIntensity['elevation'][i])
            id.append(prunedIntensity['id'][i])
            detection_score.append(prunedIntensity['detection_score'][i])
            size.append(prunedIntensity['size'][i])
            rotation.append(prunedIntensity['rotation'][i])
            velocity.append(prunedIntensity['velocity'][i])
            classification.append(prunedIntensity['classification'][i])
            P.append(self.model['Q_k'] + self.model['F_k'].dot(prunedIntensity['P'][i]).dot(np.transpose(self.model['F_k'])))   # Equation (27) in [1].

        for i in range(len(Z_k)):
            eta = []
            invSi_list = []
            S = []
            for j in range(len(prunedIntensity['w'])):
                eta.append(self.model['H_k'].dot(m[j]).astype('float64')) # Calculate predicted measurements.
                S.append(self.model['R_k'] + self.model['H_k'].dot(P[j]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
                Si = copy.deepcopy(S[j])
                invSi_list.append(np.linalg.inv(np.array(Si, dtype=np.float64)))  # Using normal inverse function
            not_associated = True
            for j in range(len(prunedIntensity['w'])):
                mean_component=eta[j]
                z_k = np.array([Z_k[i]['translation'][0],Z_k[i]['translation'][1]],dtype=object).reshape(-1,1).astype('float64')
                innovation_residual = np.array([Z_k[i]['translation'][0] - mean_component[0],Z_k[i]['translation'][1] - mean_component[1]]).reshape(-1,1).astype(np.float64)
                invSi=invSi_list[j]
                mah = np.transpose(innovation_residual).dot(invSi).dot(innovation_residual)[0][0]
                if mah < 10:
                    not_associated = False
                    break
            if not_associated == True:
                delta_x=0
                delta_y=0
                w.append(birth_rate)
                #m.append(np.array([Z_k['translation_for_all_boxes'][i][0][0], Z_k['translation_for_all_boxes'][i][1][0], Z_k['velocity_for_all_boxes'][i][0], Z_k['velocity_for_all_boxes'][i][1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
                #m.append(np.array([delta_x, delta_y, 1, 1],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.
                m.append(np.array([Z_k[i]['translation'][0]+delta_x,delta_y+Z_k[i]['translation'][1], Z_k[i]['velocity'][0], Z_k[i]['velocity'][1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.
                elevation.append(Z_k[i]['translation'][2])
                size.append(Z_k[i]['size'])
                detection_score.append(Z_k[i]['detection_score'])
                rotation.append(Z_k[i]['rotation'])
                velocity.append(Z_k[i]['velocity'])
                classification.append(Z_k[i]['detection_name'])
                max_id+=1
                id.append(max_id)
                P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.

        predictedIntensity = {}
        predictedIntensity['w'] = w
        predictedIntensity['mean'] = m
        predictedIntensity['P'] = P
        predictedIntensity['id']=id
        predictedIntensity['max_id']=max_id
        predictedIntensity['detection_score']=detection_score
        predictedIntensity['elevation']=elevation
        predictedIntensity['size']=size
        predictedIntensity['rotation']=rotation
        predictedIntensity['velocity']=velocity
        predictedIntensity['classification']=classification
        
        return predictedIntensity

    def predict_for_initial_step(self, Z_k, birth_rate):
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a gausssian component
        elevation=[]
        id=[]
        max_id=len(Z_k)-1
        size=[]
        rotation=[]
        detection_score=[]
        velocity=[]
        classification=[]

        for i in range(len(Z_k)):
            delta_x=0
            delta_y=0
            #delta_x = np.random.uniform(-10, 10)
            #delta_y = np.random.uniform(-10, 10)
            w.append(birth_rate)
            #m.append(np.array([z_k[0], z_k[1], v_init[0], v_init[1]]).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            #need to specify dtype=object for it to work on Linux, no need for Win10
            m.append(np.array([Z_k[i]['translation'][0]+delta_x, delta_y+Z_k[i]['translation'][1],Z_k[i]['velocity'][0], Z_k[i]['velocity'][1]],dtype=object).reshape(-1,1).astype('float64'))  # Target is born with [x, y, vx, vy] state format(birth at place x and y with velocity vx and vy) as mean value.
            P.append(self.model['P_k'].astype('float64'))   # Target is born with self.model['P_k'] as its variance.
            elevation.append(Z_k[i]['translation'][2])
            size.append(Z_k[i]['size'])
            detection_score.append(Z_k[i]['detection_score'])
            rotation.append(Z_k[i]['rotation'])
            velocity.append(Z_k[i]['velocity'])
            id.append(i)
            classification.append(Z_k[i]['detection_name'])

        predictedIntensity = {}
        predictedIntensity['w'] = w
        predictedIntensity['mean'] = m
        predictedIntensity['P'] = P
        predictedIntensity['id']=id
        predictedIntensity['max_id']=max_id
        predictedIntensity['detection_score']=detection_score
        predictedIntensity['elevation']=elevation
        predictedIntensity['size']=size
        predictedIntensity['rotation']=rotation
        predictedIntensity['velocity']=velocity
        predictedIntensity['classification']=classification
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
        invSi_list=[]
        for i in range(len(predictedIntensity['w'])):
            eta.append(self.model['H_k'].dot(predictedIntensity['mean'][i]).astype('float64')) # Calculate predicted measurements.
            S.append(self.model['R_k'] + self.model['H_k'].dot(predictedIntensity['P'][i]).dot(np.transpose(self.model['H_k'])).astype('float64'))  # Calculate predicted covariance matrices.
            Si = copy.deepcopy(S[i])
            
            if self.model['using_cholsky_decomposition_for_calculating_inverse_of_measurement_covariance_matrix'] == True:
                Vs = np.linalg.cholesky(np.array(Si, dtype=np.float64))
                inv_sqrt_S = np.linalg.inv(Vs)
                invSi = inv_sqrt_S.dot(np.transpose(inv_sqrt_S))
            else:
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            invSi_list.append(invSi)
            K.append(predictedIntensity['P'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P.append(predictedIntensity['P'][i] - K[i].dot(self.model['H_k']).dot(predictedIntensity['P'][i]).astype('float64'))

        constructUpdateIntensity = {}
        constructUpdateIntensity['eta'] = eta
        constructUpdateIntensity['S'] = S
        constructUpdateIntensity['K'] = K
        constructUpdateIntensity['P'] = P
        constructUpdateIntensity['invSi_list'] = invSi_list



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
        id=[]
        elevation=[]
        size=[]
        detection_score=[]
        rotation=[]
        velocity=[]
        classification=[]
        max_id=predictedIntensity['max_id']
        for i in range(len(predictedIntensity['w'])):
            # Calculate the updated weights for missed-detected targets(Only rely on the predicted result due that no measurements detected)
            w.append((1.0 - self.model['p_D'])*predictedIntensity['w'][i])
            #w.append((1.0 - predictedIntensity['detection_score'][i])*predictedIntensity['w'][i])
            # Calculate the updated mean, m, and covariance, P, for missed-detected targets(Only rely on the predicted result due that no measurements detected)
            m.append(predictedIntensity['mean'][i])
            P.append(predictedIntensity['P'][i])
            id.append(predictedIntensity['id'][i])
            detection_score.append(predictedIntensity['detection_score'][i])
            flag.append(i)
            elevation.append(predictedIntensity['elevation'][i])
            size.append(predictedIntensity['size'][i])
            rotation.append(predictedIntensity['rotation'][i])
            velocity.append(predictedIntensity['velocity'][i])
            classification.append(predictedIntensity['classification'][i])

        # "Detected targets" part of GM-PHD update. Notice this part is just mplementation of second "for loop" in step 4 in 
        # table I in original GM-PHD filter paper[1].
        # Every observation updates every target(to form all possible hypotheses for data association).
        numTargets_Jk_k_minus_1 = len(predictedIntensity['w']) # Number of Gaussian components after the prediction step
        l = 0
        clutterIntensity = self.model['clutterIntensity']
        threshold = self.model['gating_threshold']
     
        for z in range(len(Z_k)): # Iterate over m measurements/detection points
            l = l + 1   # l actually stands for l^th measurement. Thus l is always equal to z + 1.
            counter=0
            total_w_d = 0.0
            for j in range(numTargets_Jk_k_minus_1):
                mean_component=constructUpdateIntensity['eta'][j]
                z_k = np.array([Z_k[z]['translation'][0],Z_k[z]['translation'][1]],dtype=object).reshape(-1,1).astype('float64')
                innovation_residual = np.array([Z_k[z]['translation'][0] - mean_component[0],Z_k[z]['translation'][1] - mean_component[1]]).reshape(-1,1).astype(np.float64)
                #Si = copy.deepcopy(constructUpdateIntensity['S'][j])
                #invSi = np.linalg.inv(np.array(Si, dtype=np.float64)) #dim 2x2
                invSi=constructUpdateIntensity['invSi_list'][j]
                
                mah = np.transpose(innovation_residual).dot(invSi).dot(innovation_residual)[0][0]
                if mah < threshold:
                    counter+=1
                    w.append(self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j]))  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                    total_w_d+=self.model['p_D'] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j])
                    #w.append(predictedIntensity['detection_score'][j] * predictedIntensity['w'][j] * mvnpdf(z_k, constructUpdateIntensity['eta'][j], constructUpdateIntensity['S'][j]))  # Hoping multivariate_normal.pdf is the right one to use; this is for only [x, y]
                    # Calculate the updated mean, m, and updated covariance, P, for surviving targets by performing Kalman filter.
                    m.append(predictedIntensity['mean'][j] + constructUpdateIntensity['K'][j].dot(z_k - constructUpdateIntensity['eta'][j]).astype('float64'))
                    id.append(predictedIntensity['id'][j])
                    detection_score.append(Z_k[z]['detection_score'])
                    elevation.append(Z_k[z]['translation'][2])
                    size.append(Z_k[z]['size'])
                    rotation.append(Z_k[z]['rotation'])
                    velocity.append(Z_k[z]['velocity'])
                    classification.append(Z_k[z]['detection_name'])
                    P.append(constructUpdateIntensity['P'][j])

            #total_w_d = 0.0
            #for j in range(numTargets_Jk_k_minus_1):
                #total_w_d += w[l*numTargets_Jk_k_minus_1 + j]
            for j in range(counter):
                w[len(w)-j-1] = w[len(w)-j-1] / (clutterIntensity + total_w_d) # Updated weight by normalization

            #for j in range(numTargets_Jk_k_minus_1):
            #    w[l*numTargets_Jk_k_minus_1 + j] = w[l*numTargets_Jk_k_minus_1 + j] / (clutterIntensity + total_w_d) # Updated weight by normalization
        # Combine both miss-detected targets and detected targets part of the GM-PHD update.
        updatedIntensity = {}
        updatedIntensity['w'] = w
        updatedIntensity['mean'] = m
        updatedIntensity['P'] = P
        updatedIntensity['id']=id
        updatedIntensity['detection_score']=detection_score
        updatedIntensity['max_id']=max_id
        updatedIntensity['flag'] = flag
        updatedIntensity['elevation']=elevation
        updatedIntensity['size']=size
        updatedIntensity['rotation']=rotation
        updatedIntensity['velocity']=velocity
        updatedIntensity['classification']=classification


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
        id=[]
        detection_score=[]
        elevation=[]
        size=[]
        rotation=[]
        velocity=[]
        classification=[]

        # Prune out the low-weighted components, which is simple apporach to prune/truncate Gaussian components in the Gaussian miture. 
        # Only keep all the Gaussian components whose weights are larger than elim_threshold.
        Pruned_Components = [index for index,value in enumerate(updatedIntensity['w']) if value > self.model['T']]  # Indices of large enough weights
        # Pruned_Components = np.where(np.array(updatedIntensity['w']) >= self.model['T'])[0]

        # Merge the close-together components, which is simple apporach to merge, Gaussian components in the Gaussian mixture which are 
        # close to each others, together. However, there will be chance to merge two Gaussian components belong to two targets which 
        # are fairly close with each other, as one Gaussian component.
        while len(Pruned_Components) > 0:
            highWeights = np.array(updatedIntensity['w'])[Pruned_Components]
            index_of_this_component_in_pruned_list = np.argmax(highWeights)
            index_of_this_component_in_original_list = Pruned_Components[index_of_this_component_in_pruned_list]
            id.append(updatedIntensity['id'][index_of_this_component_in_original_list])
            detection_score.append(updatedIntensity['detection_score'][index_of_this_component_in_original_list])
            # Find all points with Mahalanobis distance less than merge distance threshold, U, from point updatedIntensity['mean'][j]
            Merged_Components = []  # A vector of indices of merged Gaussians.
            for iterI in range(len(Pruned_Components)):
                thisI = copy.deepcopy(Pruned_Components[iterI])
                delta_m = updatedIntensity['mean'][thisI] - updatedIntensity['mean'][index_of_this_component_in_original_list]
                mahal_dist = np.transpose(delta_m).dot(np.linalg.inv(np.array(updatedIntensity['P'][thisI],dtype=np.float64))).dot(delta_m)
                if mahal_dist <self.model['U']:
                    Merged_Components.append(thisI)  # Indices of merged Gaussians

            # The new weight of the resulted merged Guassian is the summation of the weights of the Gaussian components.
            w_bar = sum(np.array(updatedIntensity['w'])[Merged_Components])
            w.append(w_bar)

            # The new mean of the merged Gaussian is the weighted average of the merged means of Gaussian components.
            m_val = []
            for i in range(len(Merged_Components)):
                thisI = copy.deepcopy(Merged_Components[i])
                m_val.append(updatedIntensity['w'][thisI]*updatedIntensity['mean'][thisI])
            m_bar = sum(m_val)/w_bar
            m.append(m_bar.astype('float64'))
            elevation.append(updatedIntensity['elevation'][Merged_Components[-1]])
            size.append(updatedIntensity['size'][Merged_Components[-1]])
            rotation.append(updatedIntensity['rotation'][Merged_Components[-1]])
            velocity.append(updatedIntensity['velocity'][Merged_Components[-1]])
            classification.append(updatedIntensity['classification'][Merged_Components[-1]])
            # Calculating covariance P_bar is a bit trickier
            P_val = []
            for i in range(len(Merged_Components)):
                thisI = copy.deepcopy(Merged_Components[i])
                delta_m = m_bar - updatedIntensity['mean'][thisI]
                P_val.append(updatedIntensity['w'][thisI]*(updatedIntensity['P'][thisI] + delta_m.dot(np.transpose(delta_m))))

            P_bar = sum(P_val)/w_bar
            P.append(P_bar.astype('float64'))

            # Now delete the elements in Merged_Components from Pruned_Components
            for i in Merged_Components:
                Pruned_Components.remove(i)

        for idx in Pruned_Components:
            w.append(updatedIntensity['w'][idx])
            m.append(updatedIntensity['mean'][idx])
            P.append(updatedIntensity['P'][idx])
            id.append(updatedIntensity['id'][idx])
            detection_score.append(updatedIntensity['detection_score'][idx])
            elevation.append(updatedIntensity['elevation'][idx])
            size.append(updatedIntensity['size'][idx])
            rotation.append(updatedIntensity['rotation'][idx])
            velocity.append(updatedIntensity['velocity'][idx])
            classification.append(updatedIntensity['classification'][idx])


        # capping guassian components if there are more gaussians than the maximum number allowed.
        prunedMergedIntensity = {}
        
        if self.model['capping_gaussian_components'] == True:
            if len(w)>self.model['maximum_number_of_gaussian_components']:
                w_cap = []
                m_cap = []
                P_cap = []
                id_cap=[]
                detection_score_cap=[]
                elevation_cap = []
                size_cap = []
                rotation_cap = []
                velocity_cap = []
                classification_cap = []
                index_of_ranked_weights_in_ascending_order=np.argsort(w)
                index_of_ranked_weights_in_descending_order = np.flip(index_of_ranked_weights_in_ascending_order)

                for i in range(self.model['maximum_number_of_gaussian_components']):
                    idx = index_of_ranked_weights_in_descending_order[i]
                    w_cap.append(w[idx])
                    m_cap.append(m[idx])
                    P_cap.append(P[idx])
                    id_cap.append(id[idx])
                    detection_score_cap.append(detection_score[idx])
                    elevation_cap.append(elevation[idx])
                    size_cap.append(size[idx])
                    rotation_cap.append(rotation[idx])
                    velocity_cap.append(velocity[idx])
                    classification_cap.append(classification[idx])
     
                prunedMergedIntensity['w'] = w_cap
                prunedMergedIntensity['mean'] = m_cap
                prunedMergedIntensity['P'] = P_cap
                prunedMergedIntensity['id']=id
                prunedMergedIntensity['detection_score']=detection_score
                prunedMergedIntensity['max_id']=updatedIntensity['max_id']
                prunedMergedIntensity['elevation']=elevation_cap
                prunedMergedIntensity['size']=size_cap
                prunedMergedIntensity['rotation']=rotation_cap
                prunedMergedIntensity['velocity']=velocity_cap
                prunedMergedIntensity['classification']=classification_cap   
            else:
                prunedMergedIntensity['w'] = w
                prunedMergedIntensity['mean'] = m
                prunedMergedIntensity['P'] = P
                prunedMergedIntensity['id']=id
                prunedMergedIntensity['detection_score']=detection_score
                prunedMergedIntensity['max_id']=updatedIntensity['max_id']
                prunedMergedIntensity['elevation']=elevation
                prunedMergedIntensity['size']=size
                prunedMergedIntensity['rotation']=rotation
                prunedMergedIntensity['velocity']=velocity
                prunedMergedIntensity['classification']=classification         

        else:
            prunedMergedIntensity['w'] = w
            prunedMergedIntensity['mean'] = m
            prunedMergedIntensity['P'] = P
            prunedMergedIntensity['id']=id
            prunedMergedIntensity['detection_score']=detection_score
            prunedMergedIntensity['max_id']=updatedIntensity['max_id']
            prunedMergedIntensity['elevation']=elevation
            prunedMergedIntensity['size']=size
            prunedMergedIntensity['rotation']=rotation
            prunedMergedIntensity['velocity']=velocity
            prunedMergedIntensity['classification']=classification
    
        return prunedMergedIntensity

    """
    Step 6: Extracting estimated states (Multi targets state extraction)
    See table III in [1] for more info.
    """
    def extractStates(self, PrunedAndMerged):
        w = []
        m = []
        P = []
        id=[]

        elevation=[]
        size=[]
        rotation=[]
        detection_score=[]
        velocity=[]
        classification=[]
        for i in range(len(PrunedAndMerged['w'])):
            if PrunedAndMerged['w'][i] > self.model['extraction_threshold']:
                w.append(PrunedAndMerged['w'][i])
                m.append(PrunedAndMerged['mean'][i])
                P.append(PrunedAndMerged['P'][i])
                id.append(PrunedAndMerged['id'][i])
                detection_score.append(PrunedAndMerged['detection_score'][i])
                elevation.append(PrunedAndMerged['elevation'][i])
                size.append(PrunedAndMerged['size'][i])
                rotation.append(PrunedAndMerged['rotation'][i])
                velocity.append(PrunedAndMerged['velocity'][i])
                classification.append(PrunedAndMerged['classification'][i])
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
                m.append(PrunedAndMerged['mean'][max_weight_index]) # append the mean of this distribution to m
                P.append(PrunedAndMerged['P'][max_weight_index]) # append the P of this distribution to P
                PrunedAndMerged['w'][max_weight_index] = -1 # set the max weight to be 0 such that this set of data is marked as used     
        '''
        extractedStates = {}
        extractedStates['w'] = w
        extractedStates['mean'] = m
        extractedStates['P'] = P
        extractedStates['id']=id
        extractedStates['detection_score']=detection_score
        extractedStates['max_id']=PrunedAndMerged['max_id']
        extractedStates['elevation']=elevation
        extractedStates['size']=size
        extractedStates['rotation']=rotation
        extractedStates['velocity']=velocity
        extractedStates['classification']=classification

        return extractedStates
