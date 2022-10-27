"""
%% ----------------------------------- Poisson Multi-Bernoulli Mixture(PMBM) filter ------------------------------ %%
1. This Python code is reproduction for the "point target PMBM filter" originally proposed in paper [1]. 
The original Matlab code for "point target PMBM filter" could be available from authors page:
https://github.com/Agarciafernandez/MTT
2. the murty code is listed by this repository: https://github.com/erikbohnsack/murty
alternatively, the k best assignment code is offered by US Naval Resaerch lab: https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
%% ----------------------------------- Key Difference from the Matlab Code ------------------------------------------ %%
1. the matlab data structure doesn't require initialization. If a index is referenced before the prior terms are filled, matlab
will simply fill empty cells with zero. In Python we need to first specify the data structure with appropriate size, and then
referece any indexing. To circumvent this, a flag system is developped, detailed instructions will be provided duly.
2. matlab indexing start from 1, yet python starts from 0. This is quite pertinent to this project since the indexing
system is a key linkage between single target hypothesis and global hypothesis. Here we circumvent the issue by shifting
the indexing system to the left by 1 element, i.e. -1 means this track does not exist. 
Corresponding video explains MBM, PMBM, TPMBM, TPMB in detail can be seen: https://www.youtube.com/playlist?list=PLadnyz93xCLjl51PzSoFhLLSp2hAYDY0H
%% -------------------------------------------------- Reference Papers ------------------------------------------ %%
  [1] A. F. García-Fernández, J. L. Williams, K. Granström, and L. Svensson, “Poisson multi-Bernoulli mixture filter: direct 
        derivation and implementation”, IEEE Transactions on Aerospace and Electronic Systems, 2018.
  [2] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
        Processing
%% -------------------------------------------------- README ------------------------------------------------ %%

One important thing to keep in mind while reading through this document is that the procedures presented in paper [1] is only a ballpark sketch of what PMBM tries
to do, the deveils are all in the detail, some of which are clearly empirically motivated.  
First the measurements are matched with PPP components to see if it belongs to any of the previously miss detected tracks & new born PPP tracks.
If so, then a new Bernoulli track is born. The new born track index is n_previously detected track + measurement index. The measurement
associated with this track is measurement index.
Notice the birth place of a new track is based on the updated means of the underlying associated PPP components (merged if there are 
more than one associated PPP).  Although the track is indexinged based on the measurement index, and the associated measurement
is the current measurement, yet what undergirds the birth are the PPPs. 
Then, after matching with the PPP components, the measurements are matched with the Bernoulli components, under each
previous global hypothesis specified single target hypothesis.
The relasionship between global hypothesis and single target hypothesis is the following:
Global Hypotheses is a lookup table. Each row vector is a global hypothesis, each row vector is a set of possible measurement
associations for this track. The content of that lookup take is the indexing of single target hypothesis. Each single target hypothesis
would appear in multiple global hypotheses. 
The most prominent idea of PMBM is exhaustively enumerating all the possible measurements track associations, taken into account 
for mis-detection hypothesis and clutter, scoring each permutation, ranking them and then propogate the best k permutations to the next
frame. Global Hypotheses (list of lists) and single target hypotheses are simply data structures to register the permutation.
After the measurements has been matched with PPP components and Bernoulli components, a cost matrix will be generated based on the
updated data.
Notice in theory, you just need a single cost matrix. The most exhaustive form is presented in README.md. However, there
are thorny computational issues that need to be taken into consideration. Specifically, Murty algorithm can be the bottleneck
of PMBM therefore, simplied input to Murty algorithm is appreciated. In order to accomplish this, the cost matrixed is decomposed
into three parts.
1. cost_for_missed_detection_hypotheis: Missed detection is separated from the cost matrix for reasons that would be elaborated
   in deteail later.
2. cost_for_exclusive_measurement_track_associations: if a track is exclusively associated with one measurement, then there is really
   no need to put it through Murty, since there is only one option available.
3. cost_for_non_exclusive_measurement_track_associations: if a measurement is associated with more than one track, then all the
   association options need to be evaluated and ranked. This trimmed cost matrix is the only thing Murty needs.
After murty, a list of ranked association options are produced. We then need to add back the exclusive track and measurement associations
to generate a full optimal option matrix where the column vector is the measurement associated track indices. Notice this step can be thorny since it involves look up the index of measurement and track indexes.
From the full optimal option matrix, we generate the global hypotheses under this previous global hypothesis specified single
target hypothesis lookup take by converting measurement associations into its single target hypothesis index. This is the key
step for PMBM and detailed instructions would be provided duly.

%% ----------------------------------------------- Data Structure  ------------------------------------------------------------------ %%
Beware filter_pred and filter_upd are two dictionary with the following fields(Beware filter_pred has the same items as filter_upd, 
thus here we only show the examples with filter_upd):
Poisson Components:
    filter_upd['weightPois']:
        weights of the Poisson point process (PPP). It is a list with N_miss_detected_targets elements, each element is a scalar value.
    filter_upd['meanPois']:
        means of the Gaussian components of the PPP. It is a list with N_miss_detected_targets elements, each element is a vector with size (4, 1).
    filter_upd['covPois']:
        covariances of the Gaussian components of the PPP. It is a list with N_miss_detected_targets elements, each element is a matrix with size (4, 4).
MBM Components:
    filter_upd['globHyp']:
        the "matrix"(actually it is a list of list) whose number of rows is the number of global hypotheses and the number of columns is the number of 
        Bernoulli components(Note: the number of Bernoulli components here equal to "number of surviving track which were detected previously + the 
        number of new measurements", the corresponding details will be explained at "step 2: update" in the code.). Each element in a particular row 
        indicates the index of the single target hypothesis for each Bernoulli component in the particular global hypothesis. It is zero if this global 
        hypothesis does not have a particular Bernoulli component.
    filter_upd['globHypWeight']:
        the list of N_GlobalHypothese elements, each element is a scalar which is the weight of a particular global hypothesis.
    filter_upd['tracks']:
        a list of N_BernoulliComponents elements, each element is a Bernoulli component(and each Bernoulli component is actually a dictionary which contains following items).
        filter_upd['tracks'][i]:
            a dictionary contains several corresponding information of the ith Bernoulli component.
        filter_upd['tracks'][i]['track_establishing_frame']:
            a scalar value which stands for the time of birth of i-th Bernoulli component.
        filter_upd['tracks'][i]['previous_single_target_hypothesis_index']
            a dictionary for all the new single target hypotheses generated under this previous single target hypothesis
        filter_upd['tracks'][i]['meanB']:
            a list contains means of several Gaussian components corresponding to i-th Bernoulli component, each Gaussian componenet stands for each single target hypothesis corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['meanB'][j] contains the mean value of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['covB']:
            a list contains covariance matrices of several Gaussian components corresponding to i-th Bernoulli component, each Gaussian componenet stands for each single target hypothesis corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['covB'][j] contains the covariance matrix of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['eB']:
            a list contains existence probabilities of all Gaussian components(single target hypotheses) correponding to the i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['eB'][j] is a scalar value which stands for existence probability of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['measurement_association_history']:
            a list contains the history information of data associations (indices to measurements or 0 if undetected) for the all Gaussian components(all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['measurement_association_history'][j] is a list which contains the history info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
        filter_upd['tracks'][i]['measurement_association_from_this_frame']:
            a list contains the association from this frame (indices to measurements or 0 if undetected) for the all Gaussian components (all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['measurement_association_history'][j] is a list which contains the history info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
        filter_upd['tracks'][i]['weight_of_single_target_hypothesis_in_log_format']:
            a list contains the log weight information of data associations (indices to measurements or 0 if undetected) for the all Gaussian components(all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['weight_of_single_target_hypothesis_in_log_format'][j] is a list which contains the log weight info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
"""
"""
The current version of this code, is updated in 20210904.
"""


import numpy as np
import copy
import math
from murty import Murty
from util import mvnpdf, CardinalityMB
from functools import reduce
import operator
from scipy.stats import multivariate_normal


class PMBM_Filter:
    """
    PMBM Point Target Filter.
    Beware in PMBM point target filter, there are basically two types of components:
    1. Poisson Point Process/PPP component. This component is used to model "new birth targets" and "existing/surviving miss-detected targets".
    2. Multi-Bernoulli Mixture/MBM component. This component is used to model "existing/surviving detected targets".
    """
    def __init__(self, model, bayesian_filter_type, motion_model_type): 
        self.model = model # use generated model which is configured for all parameters used in PMBM filter model for tracking the multi-targets.
        """
        Key Parameters set by the model:
        
        Kalman Filter Parameters:
        model['F_k']
        model['Q_k']
        model['H_k']
        model['R_k']

        Environment Simulation Parameters
        model['xrange']
        model['yrange']
        model['clutter_intensity']
        model['p_D']
        model['p_S']
        model['m_new_birth'] dim 4x1
        model['P_newbirth'] dim 4x4
        model['number_of_new_birth_targets']
        model['number_of_new_birth_targets_init']
        model['w_birthsum']
        model['w_birthsuminit']

        Filter Parameters 
        model['maximum_number_of_global_hypotheses']
        model['T_pruning_MBM']
        model['T_pruning_Pois']
        model['eB_threshold']
        model['gating_threshold']
        model['state_extraction_option']
        model['eB_estimation_threshold']
        """
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        """
        Linear Kalman Filter:
        Guassian distributions' mean (M) and variance (P) are propogated through the linear system.
        *+ updated value
        *- predicted value

        Motion Model: F
        Motion Model Noise: Q
        Measurement Model: H
        Measurement Model Noise: R
        Innovation: S
        Kalman Gain: K

        Prediction: 
        M-_current = F * M+_previous
        P-_current = F * P+_previous * F' + Q
        
        Kalman Filter Coefficients:
        S = H * P-_current * H' + R
        K = P * H' * Inv(S)

        Update:
        M+_current = M-_current + K * (Measurement_current - H * M-_curent)
        P+_current = P-_current - K * H * P-_current
        """
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)

    """
    Step 1: Prediction. Section V-B of [1]
    For prediction step, there are three parts:
    1.1. Prediction for (existing/surviving)previously miss-detected targets(using Poisson Point Process/PPP to model).
        -- Beware Poisson Point Process/PPP is described by intensity lambda.
    1.2. Prediction for new birth targets(using Poisson Point Process/PPP to model), it will be incorporated into "prediction for miss-detected targets PPP component".
    1.3. Prediction for existing/surviving previously detected targets (Multi-Bernoulli Mixture/MBM to model).
    
    Input, filter_pruned, which is the data structure from previous frame.
    Output, filter_predicted, the data structure that containes all the predicted value.
    """
    def predict(self, filter_pruned):
        # Get data and parameters.
        F = self.model['F_k']   # Transition matrix, F.
        Q = self.model['Q_k']   # Process noise, Q.
        Ps = self.model['p_S']  # Probability of target survival, Ps.
        number_of_surviving_previously_miss_detected_targets = len(filter_pruned['weightPois'])
        number_of_new_birth_targets = self.model['number_of_new_birth_targets']
        number_of_surviving_previously_detected_targets=len(filter_pruned['tracks'])
        
        # Initate data structure for predicted step.
        # Poisson Components data structure
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        # MBM Components data structure
        if number_of_surviving_previously_detected_targets > 0:
            filter_predicted['tracks'] = [{} for i in range(number_of_surviving_previously_detected_targets)]
            filter_predicted['globHyp'] = copy.deepcopy(filter_pruned['globHyp'])
            filter_predicted['globHypWeight'] = copy.deepcopy(filter_pruned['globHypWeight'])
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                filter_predicted['tracks'][previously_detected_target_index]['eB']=[] # need to be filled in with prediction value                
                filter_predicted['tracks'][previously_detected_target_index]['meanB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['covB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'])
                filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['measurement_association_history'])
                filter_predicted['tracks'][previously_detected_target_index]['track_establishing_frame']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['track_establishing_frame'])
        else:
            filter_predicted['tracks'] = []
            filter_predicted['globHyp'] = []
            filter_predicted['globHypWeight'] = []
        """
        Step 1.1 : Prediction for surviving previously miss detected targets(i.e. the targets were undetected at previous frame and survive into current frame) by using PPP.
        """        
        # Compute where it would have been should this track have been detected in previous step.
        if number_of_surviving_previously_miss_detected_targets > 0:
            for PPP_component_index in range(number_of_surviving_previously_miss_detected_targets):
                # Get data from previous frame
                weightPois_previous = filter_pruned['weightPois'][PPP_component_index]
                meanPois_previous = filter_pruned['meanPois'][PPP_component_index]
                covPois_previous = filter_pruned['covPois'][PPP_component_index]
                
                # Compute for the prediction
                # Equation (25) in [2].
                # W-_curremt = Ps * W+_previous
                weightPois_predicted = Ps * weightPois_previous
                # Equation (26) in [2] Calculate means of the Gaussian components of miss detected targets PPP.
                # M-_current = F * M+_previous
                meanPois_predicted = F.dot(meanPois_previous)
                # Equation (27) in [1] Calculate covariance of the Gaussian component of each miss detected target PPP.
                # P-_current = F * P+_previous * F' +Q
                covPois_predicted = F.dot(covPois_previous).dot(np.transpose(F)+ Q)
                
                # Fill in the data structure
                filter_predicted['weightPois'].append(weightPois_predicted)       
                filter_predicted['meanPois'].append(meanPois_predicted) 
                filter_predicted['covPois'].append(covPois_predicted)

        """
        Step 1.2 : Prediction for new birth targets by using PPP.
        """
        # Only generate new birth target if there are no existing miss detected targets
        # Incorporate New Birth intensity into PPP. 
        for new_birth_target_index in range(number_of_new_birth_targets):
            # Compute for the birth initiation
            weightPois_birth = self.model['w_birthsum']/number_of_new_birth_targets
            meanPois_birth = np.array([self.model['m_new_birth'][0]+np.random.uniform(10), self.model['m_new_birth'][1]-np.random.uniform(10),self.model['m_new_birth'][2],self.model['m_new_birth'][3]]).reshape(-1,1).astype(np.float64)
            covPois_birth = self.model['P_new_birth']

            # Fill lin the data structure
            filter_predicted['weightPois'].append(weightPois_birth)  # Create the weight of PPP using the weight of the new birth PPP
            filter_predicted['meanPois'].append(meanPois_birth)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(covPois_birth)    # Create the variance of PPP using the variance of the new birth PPP
        """
        Step 1.3 : Prediction for existing/surviving previously detected targets(i.e. targets were detected at previous frame and survive into current frame) by using Bernoulli components, or so called Multi-Bernoulli RFS.
        """
        if number_of_surviving_previously_detected_targets > 0:
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                for single_target_hypothesis_index_from_previous_frame in range(len(filter_pruned['tracks'][previously_detected_target_index]['eB'])):
                    # Get data from previous frame
                    eB_previous = filter_pruned['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                    meanB_previous = filter_pruned['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                    covB_previous = filter_pruned['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]      
                    # Compute for the prediction
                    eB_predicted = Ps * eB_previous
                    meanB_predicted = F.dot(meanB_previous)
                    covB_predicted = F.dot(covB_previous).dot(np.transpose(F)) + Q
                    # Fill in the data structure                   
                    filter_predicted['tracks'][previously_detected_target_index]['eB'].append(eB_predicted)                    
                    filter_predicted['tracks'][previously_detected_target_index]['meanB'].append(meanB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['covB'].append(covB_predicted)
        
        return filter_predicted

    def predict_initial_step(self):
        """
        Compute the predicted intensity of new birth targets for the initial step (first frame).
        It has to be done separately because there is no input to initial step.
        There are other ways to implementate the initialization of the structure, this is just easier for the readers to understand.
        """
        # Create an empty dictionary filter_predicted which will be filled in by calculation and output from this function.
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['tracks'] = []
        filter_predicted['globHyp'] = []
        filter_predicted['globHypWeight'] = []
        # Get the parameters
        number_of_new_birth_targets_init = self.model['number_of_new_birth_targets_init']
        weightPois_initial_step = self.model['w_birthsuminit']/number_of_new_birth_targets_init
        meanPois_initial_step = np.array([self.model['m_new_birth'][0]+np.random.uniform(10), self.model['m_new_birth'][1]-np.random.uniform(10),self.model['m_new_birth'][2],self.model['m_new_birth'][3]]).reshape(-1,1).astype(np.float64)
        covPois_initial_step = self.model['P_new_birth']
        # Fill in the data structure
        for new_birth_target_index in range(number_of_new_birth_targets_init):
            filter_predicted['weightPois'].append(weightPois_initial_step)  # Create the weight of PPP using the weight of the new birth PPP
            filter_predicted['meanPois'].append(meanPois_initial_step)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(covPois_initial_step)    # Create the variance of PPP using the variance of the new birth PPP
        return filter_predicted

    """
    Step 2: Update Section V-C of [1]
    2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are still undetected at current frame, just update the weight of PPP but mean 
            and covarince remains same.
    2.2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are now associated with detections(detected) at current frame, corresponding 
            Bernoulli RFS is converted from PPP normally by updating (PPP --> Bernoulli) for each of them.
    2.2.2. For the measurements(detections) which can not be in the gating area of any previously miss detected target or any new birth target(both represented by PPP), corresponding 
            Bernoulli RFS is created by filling most of the parameters of this Bernoulli as zeors (create Bernoulli with zero existence probability, stands for detection is originated 
            from clutter) for each of them.
    2.3.1. For the previously detected targets which are now undetected at current frame, just update the eB of the distribution but mean and covarince remains same for each of them.
    2.3.2. For the previously detected targets which are now associated with detection(detected) at current frame, the parameters of the distribution is updated for each of them.  
    """
    def update(self, Z_k, filter_predicted, nth_scan):
        # Get pre-defined parameters.
        H = self.model['H_k'] # measurement model
        R = self.model['R_k'] # measurement noise
        Pd =self.model['p_D'] # probability for detection
        gating_threshold = self.model['gating_threshold']
        clutter_intensity = self.model['clutter_intensity']

        # Get components information from filter_predicted.
        number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets = len(filter_predicted['weightPois'])
        number_of_detected_targets_from_previous_frame = len(filter_predicted['tracks'])
        number_of_global_hypotheses_from_previous_frame = len(filter_predicted['globHyp'])
        number_of_measurements_from_current_frame = len(Z_k)
        
        # At the extreme case, all the measurements could be originated from previously miss-detected targets, new birth targets and clutter only, and none 
        # of the measurements originated from previously detected targets(i.e. in such extreme case, all the previously detected targets are miss-detected 
        # at current frame.)
        number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters = number_of_measurements_from_current_frame
        number_of_potential_detected_targets_at_current_frame_after_update = number_of_detected_targets_from_previous_frame + number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters

        # Initialize data structures for filter_update
        filter_updated = {}
        filter_updated['weightPois'] = []
        filter_updated['meanPois'] = []
        filter_updated['covPois'] = []

        if number_of_detected_targets_from_previous_frame==0:
            # Initiate the globHyp as all zeros which means newly generated track associate with itself
            filter_updated['globHyp']=[[int(x) for x in np.zeros(number_of_measurements_from_current_frame)]] # The global hypothesis is all associated with missed detection
            filter_updated['globHypWeight']=[1] # There would be one global hypothesis, each measurement is associated with itself.
            if number_of_measurements_from_current_frame == 0:
                filter_updated['tracks'] = [] 
            else: 
                filter_updated['tracks']=[{} for n in range(number_of_measurements_from_current_frame)] # Initiate the data structure with right size of dictionaries
                for i in range(number_of_measurements_from_current_frame): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][i]['eB']= []
                    filter_updated['tracks'][i]['covB']= []
                    filter_updated['tracks'][i]['meanB']= []
                    filter_updated['tracks'][i]['weight_of_single_target_hypothesis_in_log_format']= []
                    filter_updated['tracks'][i]['single_target_hypothesis_index_from_previous_frame']=[]
                    filter_updated['tracks'][i]['measurement_association_history']= []
                    filter_updated['tracks'][i]['measurement_association_from_this_frame']= []
                    filter_updated['tracks'][i]['track_establishing_frame'] = []

        else:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            if number_of_measurements_from_current_frame == 0:
                filter_updated['tracks'] = [] 
            else:
                filter_updated['tracks']=[{} for n in range(number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame)] # Initiate the data structure with right size of dictionaries
                # Initiate data structure for indexing 0 to number of detected target index
                for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                    number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
                    filter_updated['tracks'][previously_detected_target_index]['eB'] = []
                    filter_updated['tracks'][previously_detected_target_index]['meanB'] = []
                    filter_updated['tracks'][previously_detected_target_index]['covB'] = []
                    filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'] = []
                    filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame']=[]
                    filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'] = copy.deepcopy(filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history'])
                    filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'] = []
                    filter_updated['tracks'][previously_detected_target_index]['track_establishing_frame'] = filter_predicted['tracks'][previously_detected_target_index]['track_establishing_frame']

                # Initializing data structure for index from number of previously detected targets to number of previosly detected targets + number of measuremetns                  
                for i in range(number_of_measurements_from_current_frame): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['eB']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['meanB']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['covB']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['weight_of_single_target_hypothesis_in_log_format']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['single_target_hypothesis_index_from_previous_frame']=[]
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_history']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_from_this_frame']= []
                    filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['track_establishing_frame'] = []

        """
        Step 2.1. for update:  Missed Detection Hypothesis for PPP Components.
        Update step for "the targets which were miss detected previosly and still remain undetected at current frame, and new birth targets got undetected
        at current frame. Remain PPP. 
        """
        # Miss detected target and new birth target are modelled by using Poisson Point Process(PPP). This is the same as the miss detected target modelling part in [2].
        # Notice the reason mean and covariance remain the same is because if there is no detection, there would be no update.
        for PPP_component_index in range(number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets):
            # Get predicted data
            weightPois_predicted = filter_predicted['weightPois'][PPP_component_index]
            meanPois_predicted = filter_predicted['meanPois'][PPP_component_index]
            covPois_predicted = filter_predicted['covPois'][PPP_component_index]

            # Compute for update
            wegithPois_updated = (1-Pd) * weightPois_predicted
            meanPois_updated = meanPois_predicted # because it is miss detected, therefore, no updating required
            covPois_updated = covPois_predicted # ditto
            
            # Fill in the data structure
            filter_updated['weightPois'].append(wegithPois_updated)
            filter_updated['meanPois'].append(meanPois_updated)
            filter_updated['covPois'].append(covPois_updated)

        """
        Step 2.2. for update: Generate number_of_measurements_from_current_frame new Bernoulli components(Parts of new Bernoulli components are converted from PPP, others are 
                    created originally.). Section V-C1 of [1]
        2.2.1: Convert Poisson Point Processes to Bernoulli RFSs. Update the targets which were miss detected previosly but now get detected at current frame, by updating with 
                    the valid measurement within gating area.
        2.2.2: Create new Bernoulli RFSs. For the measurements not falling into gating area of any PPP component, it is assumed to be originated from clutter. Create a Bernoulli 
                    RSF by filling parameters with zeros for each of them anyway for data structure purpose.
        """
        for measurement_index in range(number_of_measurements_from_current_frame):    
            tracks_associated_with_this_measurement = []
            # Go through all Poisson components(previously miss-detected targets and new birth targets) and perform gating
            for PPP_component_index in range(number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets):
                mean_PPP_component_predicted = filter_predicted['meanPois'][PPP_component_index]
                cov_PPP_component_predicted = filter_predicted['covPois'][PPP_component_index]

                # Compute Kalman Filter Elements
                mean_PPP_component_measured = H.dot(mean_PPP_component_predicted).astype('float64')             
                S_PPP_component = (H.dot(cov_PPP_component_predicted).dot(np.transpose(H))+R).astype('float64') #dim 2x2
                # For numerical stability
                S_PPP_component = 0.5 * (S_PPP_component + np.transpose(S_PPP_component))
                
                ppp_innovation_residual = np.array([Z_k[measurement_index][0] - mean_PPP_component_measured[0],Z_k[measurement_index][1] - mean_PPP_component_measured[1]]).reshape(-1,1).astype(np.float64)
                Si = copy.deepcopy(S_PPP_component)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64)) #dim 2x2
                
                mahananobis_distance_between_current_measurement_and_current_PPP_component = np.transpose(ppp_innovation_residual).dot(invSi).dot(ppp_innovation_residual)[0][0]
                if mahananobis_distance_between_current_measurement_and_current_PPP_component < gating_threshold:
                    tracks_associated_with_this_measurement.append(PPP_component_index)  
            '''
            2.2.1: If current measurements is associated with PPP component(previously miss-detected target or new birth target), use this measurement to update the target, 
                    thus convert corresponding PPP into Bernoulli RFS.
            '''
            if len(tracks_associated_with_this_measurement)>0: # If there are PPP components could be associated with current measurement.
                meanB_sum = np.zeros((len(H[0]),1))
                covB_sum = np.zeros((len(H[0]),len(H[0])))
                weight_of_true_detection = 0
                for associated_track_index in tracks_associated_with_this_measurement:
                    # Get the predicted data
                    mean_associated_track_predicted = filter_predicted['meanPois'][associated_track_index]
                    cov_associated_track_predicted = filter_predicted['covPois'][associated_track_index]
                    weight_associated_track_predicted = filter_predicted['weightPois'][associated_track_index]
                    
                    mean_associated_track_measured = H.dot(mean_associated_track_predicted).astype('float64')                     
                    # Compute for innovation covariance S: H * P-_current * H' + R
                    S_associated_track = (H.dot(cov_associated_track_predicted).dot(np.transpose(H))+R).astype('float64')
                    # For numerical stability
                    S_associated_track = 0.5 * (S_associated_track + np.transpose(S_associated_track))
                    
                    #Si = copy.deepcopy(S_associated_track)
                    #invSi = np.linalg.inv(np.array(Si, dtype=np.float64))

                    # Empirically this method is numerically more stable
                    Vs= np.linalg.cholesky(S_associated_track) 
                    Vs = np.matrix(Vs)
                    #log_det_S_pred_j= 2*np.log(reduce(operator.mul, np.diag(Vs))) 
                    Si = copy.deepcopy(Vs)
                    inv_sqrt_Si = np.linalg.inv(np.array(Si, dtype=np.float64))
                    invSi= inv_sqrt_Si*np.transpose(inv_sqrt_Si)

                    # Compute for Kalman Gain K: P * H' * Inv(S)
                    K_associated_track = cov_associated_track_predicted.dot(np.transpose(H)).dot(invSi).astype('float64')
                    track_innovation_residual = np.array([Z_k[measurement_index][0] - mean_associated_track_measured[0],Z_k[measurement_index][1] - mean_associated_track_measured[1]]).reshape(-1,1).astype(np.float64)
                    
                    # Compute for update
                    # M+_current = M-_current + K * (Measurement_current - H * M-_curent)
                    mean_associated_track_updated = mean_associated_track_predicted + K_associated_track.dot(track_innovation_residual) # it is a column vector with lenghth 4
                    # P+_current = P-_current - K * H * P-_current
                    cov_associated_track_updated = cov_associated_track_predicted - K_associated_track.dot(H).dot(cov_associated_track_predicted).astype('float64')
                    # For numerical stability
                    cov_associated_track_updated = 0.5 * (cov_associated_track_updated + np.transpose(cov_associated_track_updated))               
                    
                    # Update the target by using current measurement, convert PPP into Bernoulli RFS(currently detected target is represented using Bernoulli).
                    # weight = Probability_of_detection * weight_previous_frame * measurement_likelihood
                    # this is according the the equation 45 of [1]
                    weight_for_track_detection = Pd*weight_associated_track_predicted*mvnpdf(Z_k[measurement_index], mean_associated_track_measured,S_associated_track)
                    # according to equation 45 of [1]
                    weight_of_true_detection += weight_for_track_detection
                    meanB_sum += weight_for_track_detection*(mean_associated_track_updated)
                    covB_sum += weight_for_track_detection*cov_associated_track_updated + weight_for_track_detection*(mean_associated_track_updated.dot(np.transpose(mean_associated_track_updated)))

                # Gaussian mixture reduction
                # If current measurement is associated with more than one targets(PPP components, now already converted to Bernoullu components), merging all the guassians.
                # Notice the birth place of a new track is based on the updated means of the underlying associated PPP components (merged if there are 
                # more than one associated PPP).  Although the track is indexinged based on the measurement index, and the associated measurement
                # is the current measurement, yet what undergirds the birth are the PPPs. 
                meanB_updated = meanB_sum/weight_of_true_detection
                covB_updated = covB_sum/weight_of_true_detection - (meanB_updated*np.transpose(meanB_updated))
                # the existence probability is weightB_sum/probability_of_detection
                # the relationship between existence probability and single target hypothesis probability is 
                # presented in equation 39 of [1]
                # here probability_of_detection * eB_updated = probability of true detection.
                # this is a relationship that would be uphold throughout this document.
                probability_of_detection = weight_of_true_detection + clutter_intensity
                # potentially, this should be division instead of summation
                #probability_of_detection = weight_of_true_detection/clutter_intensity
                eB_updated = weight_of_true_detection/probability_of_detection
                # Fill in the data structure
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated) # Notice the meaning of of eB, is for cardinality computation, weightB is the hypothesis probability, which are two different things
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection)) # weightB is used for cost matrix computation
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(measurement_index) # register history 
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(measurement_index) # register history 
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['track_establishing_frame']=nth_scan # the track is initiated at time nth_scan

            else: 
                '''
                2.2.2
                If there is not any PPP component(previously miss-detected target or new birth target) could be associated with current measurement, assume this measurement is originated from clutter. 
                We still need to create a Bernoulli component for it, since we need to guarantee that ever measurement generate a Bernoulli RFS.
                The created Bernoulli component has existence probability zero (denote it is clutter). It will be removed by pruning.
                '''
                meanB_updated = mean_PPP_component_predicted
                covB_updated = cov_PPP_component_predicted
                # weight_of_true_detection = 0 therefore existence probability is 0
                probability_of_detection = clutter_intensity #This measurement is a clutter
                eB_updated = 0

                # in the global hypothesis generating part of the code, this option would be registered as h_-1 this track does not exist
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection))
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated) # This can be set to zero, essentially, it doesn't matter
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated) # ditto
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1) #-1 means this track does not exist
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['track_establishing_frame'].append(nth_scan) # If it is a potential detection, then it would start from Nth scan.

        """
        Step 2.3. for update: Section V-C2 of [1]
        Update for targets which got detected at previous frame.
        """
        for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
            
            number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
            filter_updated['tracks'][previously_detected_target_index]['track_establishing_frame'] = copy.deepcopy(filter_predicted['tracks'][previously_detected_target_index]['track_establishing_frame'])
        
            # Loop through all single target hypotheses belong to global hyptheses from previous frame. 
            for single_target_hypothesis_index_from_previous_frame in range(number_of_single_target_hypotheses_from_previous_frame):
                # Get the data from filter_predicted
                mean_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                cov_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                eB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                #weight_single_target_hypothesis_predicted_in_log_format = filter_predicted['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][single_target_hypothesis_index_from_previous_frame]
                """
                Step 2.3.1. for update: Undetected Hypothesis
                Update the targets got detected previously but get undetected at current frame.
                """
                # Compute for Missed detection Hypothesis
                probability_for_track_exist_but_undetected = eB_single_target_hypothesis_predicted*(1-Pd)
                probability_for_track_dose_not_exit = 1-eB_single_target_hypothesis_predicted
                eB_undetected = probability_for_track_exist_but_undetected/(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit)
                # this is according to page 10 of [1] 
                # it is not computed strictly according to the equation, weight_single_target_hypothesis_predicted_in_log_format
                # is omitted for reasons presented at the bottom right corner of page 10.
                # Note that we normalise the previous weights by ρj,i (∅) so that the weight 
                # of a hypothesis that does not assign a measurement to a target is the same for 
                # an old and a new target. This is just done so that we can obtain the k-best 
                # global hypotheses efficiently using Murty’s algorithm but we do not alter the 
                # real weights, which are unnormalised. 
                # notice the relationship between wij and eB
                # weight_of_single_target_hypothesis_undetected * eB_undetected = wij*eB_previous*(1-Pd)
                # weight and eB essentially decompose a probability and propagate separately.
                # in PHD and CPHD, the weight is simply probability for track exist but undetected, no decomposition required.
                # in PMBM weight is the cost for the association between track and measurement
                # eB is utilized during the pruning stage for track death.
                # track is marked dead if all the single target hypothesis associated with this track
                # has eB that is smaller than the eB threshold.
                # notice if the single target hypothesis from previous frame as eB of 1, here eB_undetected would still be 1.
                # the difference will be that weight is now (1-Pd) which can be an exceedingly small number, resulting
                # in large cost and therefore will not be chosen. 
                weight_of_single_target_hypothesis_undetected_in_log_format = np.log(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit) # does not exist plus exist but not detected, how likely is this hypothesis
                #weight_of_single_target_hypothesis_undetected_in_log_format = weight_single_target_hypothesis_predicted_in_log_format + np.log(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit) # does not exist plus exist but not detected, how likely is this hypothesis
                # mean and cov would remain the same since there is no measurement associated with it.
                mean_single_target_hypothesis_undetected = mean_single_target_hypothesis_predicted
                cov_single_target_hypothesis_undetected = cov_single_target_hypothesis_predicted
                
                # Fill in the data structure
                filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_undetected)
                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_undetected_in_log_format)
                filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(-1) # -1 as the index for missed detection
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(-1) # -1 as the index for missed detection
                """
                Step 2.3.2. for update:
                Update the targets got detected previously and still get detected at current frame.
                Beware what we do here is to update all the possible single target hypotheses and corresponding cost value for every single target hypothesis(each 
                target-measurement possible association pair). The single target hypothese which can happen at the same time will form a global hypothesis(joint 
                event), and all the global hypotheses will be formed exhaustively later by using part of "all the possible single target hypotheses". 
                """
                
                # Compute Kalman Filter Elements
                # Compute for innovation S: H * P-_current * H' + R
                S_single_target_hypothesis = (H.dot(cov_single_target_hypothesis_predicted).dot(np.transpose(H))+R).astype('float64')
                # For numerical stability
                S_single_target_hypothesis = 0.5 * (S_single_target_hypothesis + np.transpose(S_single_target_hypothesis))
                mean_single_target_hypothesis_measured = H.dot(mean_single_target_hypothesis_predicted).astype('float64')
                
                # alternatively, seems that inverse can be computed this way. 
                #Si = copy.deepcopy(S_single_target_hypothesis)
                #invSi = np.linalg.inv(np.array(Si, dtype=np.float64))

                # Garsia's method
                # empirically, this is more stable
                Vs= np.linalg.cholesky(S_single_target_hypothesis) 
                Vs = np.matrix(Vs)
                #log_det_S_pred_j= 2*np.log(reduce(operator.mul, np.diag(Vs)))
                Si = copy.deepcopy(Vs)
                inv_sqrt_Si = np.linalg.inv(np.array(Si, dtype=np.float64))
                invSi= inv_sqrt_Si*np.transpose(inv_sqrt_Si)

                # Compute for Kalman Gain K: P * H' * Inv(S)
                K_single_target_hypothesis = cov_single_target_hypothesis_predicted.dot(np.transpose(H)).dot(invSi).astype('float64')
                
                for measurement_index in range(number_of_measurements_from_current_frame): # starting from m_1, since m_0 means missed detection
                    detected_track_innovation_residual = np.array([Z_k[measurement_index][0] - mean_single_target_hypothesis_measured[0],Z_k[measurement_index][1] - mean_single_target_hypothesis_measured[1]]).reshape(-1,1).astype(np.float64)
                    mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement = np.transpose(detected_track_innovation_residual).dot(invSi).dot(detected_track_innovation_residual)[0][0]
                    if mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement < gating_threshold: 
                        # Perform Kalman update with this measurement
                        # M+_current = M-_current + K * (Measurement_current - H * M-_curent)
                        mean_single_target_hypothesis_updated = mean_single_target_hypothesis_predicted + K_single_target_hypothesis.dot(detected_track_innovation_residual) # it is a column vector with lenghth 4
                        # P+_current = P-_current - K * H * P-_current
                        cov_single_target_hypothesis_updated = cov_single_target_hypothesis_predicted - K_single_target_hypothesis.dot(H).dot(cov_single_target_hypothesis_predicted).astype('float64')
                        # For numerical stability
                        cov_single_target_hypothesis_updated = 0.5 * (cov_single_target_hypothesis_updated + np.transpose(cov_single_target_hypothesis_updated))
                        # notice that follow previous logic:
                        # w could be (probability_for_track_exist_and_detected+probability_for_track_dose_not_exist_but_detected)
                        # w = w* [eB*Pd*measurement_liklihood+(1-eB)*clutter_intensity] (does not exist and detected by clutter is probability of a joint event)
                        # eB should be probability for track exist and detected/(probability_for_track_exist_and_detected+probability_for_track_dose_not_exist_but_detected)
                        # eB = eB*Pd/(eB*Pd+(1-eB)*clutter_intensity)
                        # therefore w * eB = probability for track exist and detected
                        # things can be defined as the following, if strictly follow the logic presented in the paper.
                        # weight_of_single_target_hypothesis_updated_in_log_format = np.log(Pd*eB_single_target_hypothesis_predicted*mvnpdf(Z_k[measurement_index], mean_single_target_hypothesis_measured,S_single_target_hypothesis)+(1-eB_single_target_hypothesis_predicted)*clutter_intensity)
                        # eB_single_target_hypothesis_updated = Pd*eB_single_target_hypothesis_predicted*mvnpdf(Z_k[measurement_index], mean_single_target_hypothesis_measured,S_single_target_hypothesis)/(Pd*eB_single_target_hypothesis_predicted*mvnpdf(Z_k[measurement_index], mean_single_target_hypothesis_measured,S_single_target_hypothesis)+(1-eB_single_target_hypothesis_predicted)*clutter_intensity)
                        # this is unnecessary since (1-Pd)*clutter_intensity is an exceedingly small number 
                        # computationally it is just easier to set eB as 1.
                        # 
                        # Another important implementation detail here is that notice it is did not follow the definition
                        # presented at equation in page 10 of [1] this is simply w_ij*e_ij*measurement_likelihood.      
                        # weight_of_single_target_hypothesis_updated_in_log_format = weight_single_target_hypothesis_predicted_in_log_format + np.log(Pd * eB_single_target_hypothesis_predicted * mvnpdf(Z_k[measurement_index], mean_single_target_hypothesis_measured,S_single_target_hypothesis))
                        # this is because in order to fairly comparing PPP and Bernoulli components with Murty
                        # the costs are normalized against the weight of mis-detection hypothesis. 
                        # for detailed reasoing of this implementation detail, please refer the the last paragraph of Page 10 of [1].
                        # because mis-detection hypothesis are divided away, therefore, it is unnecessary to add weight_single_target_hypothesis_predicted_in_log_format here.

                        # weight is presented in its log format in order to accenturate the difference. While in theory, Murty can be sensative enough
                        # to disern the optimal option with dicimal input, but given the numerical error, it is better to err on the side of causion
                        # and convert things into log format such that minute discrepencies in dicimal system can be accentuated, yielding more reliable outcome.                  
                        
                        # page 10 of [1], bottom left corner
                        weight_of_single_target_hypothesis_updated_in_log_format =np.log(Pd * eB_single_target_hypothesis_predicted * mvnpdf(Z_k[measurement_index], mean_single_target_hypothesis_measured,S_single_target_hypothesis))
                    
                        #filter_upd.tracks{i}.weightBLog_k(index_hyp)=log(eB_j*p_d)+quad-1/2*log_det_S_pred_j-Nz*log(2*pi)/2 ;
                        #weight_of_single_target_hypothesis_updated_in_log_format = np.log(Pd * eB_single_target_hypothesis_predicted)+0.5*mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement-0.5*log_det_S_pred_j-2*np.log(2*math.pi)/2 
                        #position = [Z_k[measurement_index][0][0],Z_k[measurement_index][1][0]]
                        #mean = [mean_single_target_hypothesis_measured[0][0], mean_single_target_hypothesis_measured[1][0]] 
                        #weight_of_single_target_hypothesis_updated_in_log_format =np.log(Pd * eB_single_target_hypothesis_predicted * multivariate_normal.pdf(position, mean ,S_single_target_hypothesis))

                        eB_single_target_hypothesis_updated = 1
                        # Fill in the data structure
                        filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_single_target_hypothesis_updated) 
                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_updated_in_log_format)
                        filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(measurement_index) # Add current measurement to association history
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(measurement_index) # Add current measurement to association history

        """
        Step 2.4. for update:
        Update Global Hypotheses as described in section V-C3 in [1].
        The objective of global hypotheses is to select k optimal single target hypotheses to propogate towards the next step.
        """

        if number_of_detected_targets_from_previous_frame>0:

            weight_of_global_hypothesis_in_log_format=[]
            globHyp=[]
            '''
            This step is similar to the joint_association_event_matrix genration in JPDA. The objective is to
            exclude all double association, i.e. a measurement is associated with more than one targets since this
            clearly violates the single target assumption for multi-object tracking. Notice the presentation of this information is different to that of JPDA. 
            in JPDA, every association event is presented by a hot-hit matrix, where one means an established association, 
            0 means no association. Yet here in PMBM, instead of binary registration, the information in registered in a much more 
            succinct manner with its single target hypothesis indexing. Global hypothesis matrix contains the same inforamtion as that of the list of association matrices in JPDA.
            filter_updated['tracks'] is a dictionary of all the possible single target associations for each target
            but in order to assemble the new global hypotheses based on those single target hypotheses, new global constraints are applied. 
            Specifically, the global constraints is the following:
            global hypotheses are a collection of these single-target hypotheses, with the conditions that 
            no measurement is left without being associated and a measurement can only be assigned to 
            one single target hypothesis. 
            The global hypotheses matrix and its corresponding global hypotheses weight matrix is of size:
            (number of global hypotheses, number of potential tracks after update)
            in this matrix, each element is the index of single target hypotheis. 
            For instance, h_-1 means this track does not exist, h_0 means this track has index (n_previous tracks + measurement index) and the associated measurement is measurement index.
            the format of one specific global hypothesis could be: [2,1,3,1,0,1,2]. The numerical value represent the indexing of single target hypothesis
            Please note that the indexing is not unique. THIS CAN BE A POINT OF CONFUSION. the indexing of single target hypothesis is track specific, 
            therefore, without double association doesn't mean the global hypothesis is all unique indexing. 
            
            ABOUT THE VARIABLE NAME:
            
            cost_matrix_log: is the matrix with weight_log filled in. (n_previously_detected_tracks + n_measurements) * n_measurement 
            weight_for_missed_detection_hypotheses: is the list for the missed detection hypothesis associated with each track
            cost_matrix_log_exclusive: is the matrix for track and measurement with exclusive association
            cost_matrix_log_non_exclusive: multiple: is the matrix for track and measurement with non-exclusive association
            
            indices_of_measurements_exclusive: the indices of measurements with exclusive track association.
            indices_of_tracks_exclusive: the indices of tracks with exclusive measurement association.
            indices_of_measurements_non_exclusive: the indices of measurements with non-exclusive track association.
            indices_of_tracks_non_exclusive: the indices of tracks with non-exclusive track association.

            optimal_associations_non_exclusive: the optimal associatioins for non-exclusively paired track and measurements.
            optimal_associations_all: the optimal associations for all tracks and measurements.
            cost_for_optimal_associations_non_exclusive: is a list of costs for all the association options.                       
            '''
            for global_hypothesis_index_from_pevious_frame in range(number_of_global_hypotheses_from_previous_frame):
                '''
                Step 2.4.1 Generate Cost Matrix For Each Global Hypothesis Index from previous frame
                '''
                # Initiate a data structure for each global hypothesis.
                # cost matrix has n_track + n_measurement rows and n_measurement columns.
                # the data filled in is the weight of each single target hypothesis in log format..
                cost_matrix_log=-np.inf*np.ones((number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame , number_of_measurements_from_current_frame))
                # weight_for_missed_detection_hypotheses has 1 row n_measurements columns. 
                # the data filled in is the weight of miss detection hypothesis
                weight_for_missed_detection_hypotheses=np.zeros(number_of_detected_targets_from_previous_frame)
                # optimal assocaition matrix, has at most max_number_of_global_hypothese rows
                # and n_measurement columns.
                optimal_associations_all = []
                '''
                Step 2.4.1.1 Fill in cost_matrix with regard to the detected tracks.
                We require three flags from filter_updated data structure in order to fill in the cost matrix.
                1. filter_updated['tracks'][track_index][weight_of_single_target_hypothesis_in_log_format] the actual data to fill in the cost matrix
                2. filter_updated['tracks'][track_index][single_target_hypothesis_index_from_previous_frame] the flag indicate under which single target hypothesis of previous frame was this new single target hypothesis generated
                3. filter_updated['tracks'][track_index]['measurement_association_from_this_frame'] which measurement was associated with this track in this single target hypothesis                
                '''
                for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                    '''
                    1. read out the previous single target hypothesis index speficied by the global hypothesis
                    '''
                    single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index_from_pevious_frame][previously_detected_target_index] # Hypothesis for this track in global hypothesis i 
                    if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis != -1: # if this track exist                                
                        # Fill in the cost matrix
                        # Only get data that is generated under the corresponding single_target_hypothesis_index_from_previous_frame
                        '''
                        2. get the indices of current single target hypotheses generated under this global hypothesis 
                        '''
                        new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis = [idx for idx, value in enumerate(filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame']) if value == single_target_hypothesis_index_specified_by_previous_step_global_hypothesis]
                        #if len(new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis)>0:
                        # missed detection hypothesis is the first data generated under any previous single target hypothesis
                        '''
                        3. fill in the cost_detection with the weight of missed detection under this 
                        '''
                        missed_detection_hypothesis_weight = filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[0]]
                        # fill in the weight_for_missed_detection_hypotheses list with mis-detection weight
                        weight_for_missed_detection_hypotheses[previously_detected_target_index]=missed_detection_hypothesis_weight
                        # Get the track measurement association from data structure
                        measurement_association_list_generated_under_this_previous_global_hypothesis = [filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'][x] for x in new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[1:]]
                        if len(measurement_association_list_generated_under_this_previous_global_hypothesis) >0:
                            for idx, associated_measurement in enumerate(measurement_association_list_generated_under_this_previous_global_hypothesis):
                                idx_of_current_single_target_hypothesis = new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[idx+1]
                                # remove the weight of the misdetection hypothesis to use Murty 
                                # for detailed explanation as of why this step is necessary, please refer to page 10 of [1], at the bottom left conner.
                                # notice the mis-detection hypothesis is subtracted away here
                                # because in order to fairly compare PPP and Bernoulli components with murty, 
                                # the weight of Bernoulli is first normlized by bench marking against the mis-detection hypothesis.

                                #  Note that we normalise the previous weights by ρj,i (∅) so that the weight of a hypothesis that does 
                                # not assign a measurement to a target is the same for an old and a new target. This is just done so that we 
                                # can obtain the k-best global hypotheses efficiently using Murty’s algorithm but we do not alter the real weights, 
                                # which are unnormalised. 
                                cost_matrix_log[previously_detected_target_index][associated_measurement] = \
                                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][idx_of_current_single_target_hypothesis]\
                                -missed_detection_hypothesis_weight
                '''
                Step 2.4.1.2 Fill in the cost matrix with regard to the newly initiated tracks
                '''
                for measurement_index in range(number_of_measurements_from_current_frame):
                    cost_matrix_log[number_of_detected_targets_from_previous_frame+measurement_index][measurement_index]=filter_updated['tracks'][number_of_detected_targets_from_previous_frame+measurement_index]['weight_of_single_target_hypothesis_in_log_format'][0]

                '''
                Step 2.4.2 Genereta the K (which varies each frame) optimal option based on cost matrix
                1. Remove -Inf rows and columns for performing optimal assignment. We take them into account for indexing later. Columns that have only one value different from Inf are not fed into Murty either.
                2. Use murky to get the kth optimal options
                3. Add the cost of the misdetection hypothesis to the cost matrix
                4. Add back the removed infinity options
                '''
                # generate the one hit matrix where only non-infinity elements are registered as 1
                # the dimmention of cost matrix (old_track+potential_new_tracks, measurements)
                indices_of_cost_matrix_with_valid_elements = 1 - np.isinf(cost_matrix_log)
                # keep the columns whose sum is greater than 1, which indicate that there are more than one valid associations
                indices_of_measurements_non_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])>1]
                if len(indices_of_measurements_non_exclusive)>0:
                    # get the indices of tracks with non-exclusive associations
                    indices_of_tracks_non_exclusive=[x for x in range(len(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive]))) if sum(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive])[x]>0)]
                    # slice for cost matrix with non_exclusive assocations
                    # the is going to the be matrix for Murty algorith
                    cost_matrix_log_non_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_non_exclusive]))[indices_of_tracks_non_exclusive]
                else:
                    indices_of_tracks_non_exclusive = []
                    cost_matrix_log_non_exclusive = []
                
                # if the column vector sums to one, then this column is to stay for certain
                indices_of_measurements_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])==1]
                if len(indices_of_measurements_exclusive) > 0:
                    # get the indices of tracks with exclusive measurement association by counting the one-hit valid matrix
                    indices_of_tracks_exclusive = [np.argmax(indices_of_cost_matrix_with_valid_elements[:,x]) for x in indices_of_measurements_exclusive]
                    # slice for the cost matrix with exclusive measurement and track associations.
                    # this is going to be the matrix to add to the result of murty.
                    cost_matrix_log_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_exclusive]))[indices_of_tracks_exclusive]
                else:
                    indices_of_tracks_exclusive = []
                    cost_matrix_log_exclusive = []
                
                # if there are no measurements that is associated with multiple tracks
                if len(cost_matrix_log_non_exclusive)==0:
                    association_vector=np.zeros(number_of_measurements_from_current_frame)
                    for index_of_idx, idx in enumerate(indices_of_measurements_exclusive):
                        association_vector[idx]=indices_of_tracks_exclusive[index_of_idx]
                    optimal_associations_all.append(association_vector)
                    # here because the optimal_association_non_exlusive is empty matrix
                    # the cost associated with empty list would be 0
                    cost_for_optimal_associations_non_exclusive = [0]

                else:
                    # Number of new global hypotheses from this global hypothesis
                    # For global hypothesis j, whose weight is wj ∝ ∏n i=1 wj,i, we suggest choosing k = dNh ·wje,
                    # where it is assumed that we want a maximum number Nh of global hypotheses as in [21]. This way, global hypotheses
                    # with higher weights will give rise to more global hypotheses. 
                    # We run Murty algorithm to select only k best(beware the len(opt_indices) equals to k) global hypotheses among all global hypotheses. We call the function by using transpose and negative value
                    # murty is compiled cpp file. For usage example, please refer to:
                    # https://github.com/erikbohnsack/murty/blob/master/src/murty.hpp
                    # Murty would provide the optimal choices for each row, under the constraints that each column can only provide at most one element.
                    # cost_matrix has potential number of tracks row and number of measurements column. 
                    # trasnpose(cost_matrix) has number of measurements rows, and number of potential tracks column.
                    # the output would be a row vector, each row(measurements) is associated with one track
                    # ascending order of cost, therefore, the lower the cost the better it is.
                    k_best_global_hypotheses_under_this_previous_global_hypothesis=np.ceil(self.model['maximum_number_of_global_hypotheses']*filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame])
                    cost_matrix_object_went_though_murty = Murty(-np.transpose(cost_matrix_log_non_exclusive)) 
                    optimal_associations_non_exclusive = []
                    cost_for_optimal_associations_non_exclusive = []

                    # notice that k_best might be larger than the maximum number of options generated by murty
                    # therefore, use still valid flag to stop if we have come to the end of all the options.
                    for iterate in range(int(k_best_global_hypotheses_under_this_previous_global_hypothesis)):
                        still_valid_flag,ith_optimal_cost,ith_optimal_solution = cost_matrix_object_went_though_murty.draw()
                        if still_valid_flag == True:
                            # each measurement is either associated with an old track or a new track
                            optimal_associations_non_exclusive.append(ith_optimal_solution) # each solution is a list of row indices
                            cost_for_optimal_associations_non_exclusive.append(ith_optimal_cost) # should be of ascending order of cost
                        else:
                            break
                    # Optimal indices without removing Inf rows
                    # the exclusive associations are placed at its appropriate position.
                    # the indices of measurements non exclusive/exclusive are the lookup take for this procedure
                    optimal_associations_all = -np.inf*np.ones((len(optimal_associations_non_exclusive),number_of_measurements_from_current_frame))           
                    for ith_optimal_option_index, ith_optimal_association_vector in enumerate(optimal_associations_non_exclusive):
                        # First handle the case where there are duplicated associations
                        for idx_of_non_exclusive_matrix, ith_optimal_track_idx in enumerate(ith_optimal_association_vector):
                            # find out the actual index through the lookup vectors
                            actual_measurement_idx = indices_of_measurements_non_exclusive[idx_of_non_exclusive_matrix]
                            actual_track_idx = indices_of_tracks_non_exclusive[ith_optimal_track_idx]
                            optimal_associations_all[ith_optimal_option_index][actual_measurement_idx]=actual_track_idx
                        # Then handle the case wehre there are single association
                        for idx_of_exclusive_matrix, actual_measurement_idx in enumerate(indices_of_measurements_exclusive):
                            actual_track_idx = indices_of_tracks_exclusive[idx_of_exclusive_matrix]
                            optimal_associations_all[ith_optimal_option_index][actual_measurement_idx]= actual_track_idx
                
                # Compute for cost fixed from cost matrix exclusive assocition.
                # this need to be added back to the weight of global hypothesis because ealier, we deleted this part out.
                weight_of_exclusive_assosications = 0
                for row_index in range(len(cost_matrix_log_exclusive)):
                    weight_of_exclusive_assosications += cost_matrix_log_exclusive[row_index][row_index]
                
                # This part follow this detail described at page 10 of [1]
                # mis-detection hypothesis was first divided (the log subtracted) and the added back here.
                # Note that we normalise the previous weights by ρj,i (∅) so that the weight of a hypothesis that does 
                # not assign a measurement to a target is the same for an old and a new target. 
                # This is just done so that we can obtain the k-best global hypotheses efficiently using Murty’s 
                # algorithm but we do not alter the real weights, which are unnormalised. 

                # notice np.log(filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame]))
                # is used here instead of sum(log(wji)) because of the equation presented in the upper left corner
                # on page 11 of [1], the global hypothesis weight is propotional to multiplication of wji
                # the coefficient is the normalization term, because global hypothese need to be normalized.

                # here not only should mis-detection hypothesis be added back, 
                # but because the computation of mis-detection term did not follow the formular strictly.
                # the wji_predicted was omitted, but need to be added back here.
                # this term ranges from 0.5 to 10, so it is quite important to be added back properly.
                for ith_optimal_option in range(len(optimal_associations_all)):
                    # wj_new = wj_old* all wji made up of this global hypothesis
                    weight_of_global_hypothesis_in_log_format.append(-cost_for_optimal_associations_non_exclusive[ith_optimal_option]+np.sum(weight_for_missed_detection_hypotheses)+weight_of_exclusive_assosications+np.log(filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame])) # The global weight associated with this hypothesis

                '''
                Step 2.4.3 Generate the new global hypothesis based on the cost matrix of this previous global hypothesis
                '''
                # Initiate Global Hypoethesis data structure
                globHyp_from_current_frame_under_this_globHyp_from_previous_frame=np.zeros((len(optimal_associations_all), number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame))
                '''
                please refer to figure 2 of [1]. The following part discribes track 1 and 2 which is an
                established Bernoulli track. 

                For established Bernoulli tracks, the existance probability is always 1. 
                The hypothesis indexing is the following:
                h_* associated with measurement with indexing *, for instance h_2 means associated with measurement1, notice it is very important to get the indexing of measurement right.
                '''
                for track_index in range(number_of_detected_targets_from_previous_frame): # first  update the extablished Bernoulli components
                    single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index_from_pevious_frame][track_index] # Readout the single target hypothesis index as specified by the global hypothesis of previous step
                    # of all the single target hypotheses associated for this track
                    # only extract the index of single target hypotheses that is generated under
                    # single_target_hypothesis_index_specified_by_previous_step_global_hypothesis by reading out the single_target_hypothesis_index_from_previous_frame flag
                    indices_of_new_hypotheses_generated_from_this_previous_hypothesis = [idx for idx, value in enumerate(filter_updated['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame']) if value == single_target_hypothesis_index_specified_by_previous_step_global_hypothesis]
                    # If under this previous global hypothesis specified single target hypothesis
                    # there is one new single target  hypothesis
                    # then this is only the missed detection hypothesis
                    for ith_optimal_option_index, ith_optimal_option_measurement_track_association_vector in enumerate(optimal_associations_all): 
                        # if under this global hypothesis, measurement is associated with this track
                        indices_of_ith_optimal_option_associated_measurement_list = [idx for idx, value in enumerate(ith_optimal_option_measurement_track_association_vector) if value == track_index] # if this track is part of optimal single target hypothesis          
                        if len(indices_of_ith_optimal_option_associated_measurement_list)==0: # if under this global hypothesis, this track is not associated with any measurement, therefore a missed detection
                            if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis == -1:
                                # if the single target hypothesis specified by previous step global hypothesis is -1
                                # then pass the value to the single target hypothesis of this frame
                                # -1 means this track does not exist
                                single_target_hypothesis_index = single_target_hypothesis_index_specified_by_previous_step_global_hypothesis
                            else:
                                # if the single target hypothesis speficied by previous global hypothesis is not -1
                                # and there is no measurement associated with this track
                                # then the single target hypothesis is the one indexing mis-detection
                                single_target_hypothesis_index = indices_of_new_hypotheses_generated_from_this_previous_hypothesis[0]
                        else:
                            # if there are measurement associated with this track
                            # notice len(indices_of_new_hypotheses_generated_from_this_previous_hypothesis) can at most be one
                            # because after Murty, there are only one measurement associated with the track
                            ith_best_optimal_measurement_association_for_this_track = indices_of_ith_optimal_option_associated_measurement_list[0]
                            # for every single target hypothesis generated under this global hypothesis of previous frame
                            for new_single_target_hypothesis_index in indices_of_new_hypotheses_generated_from_this_previous_hypothesis[1:]:
                                # if the measurement associated with this track is the ith best optimal measurement
                                if filter_updated['tracks'][track_index]['measurement_association_from_this_frame'][new_single_target_hypothesis_index]==ith_best_optimal_measurement_association_for_this_track:
                                    # then the single target hypothesis frame is set to be the index of this new_single_target_hypothesis_index
                                    single_target_hypothesis_index=new_single_target_hypothesis_index
                        # fill in the data structure
                        globHyp_from_current_frame_under_this_globHyp_from_previous_frame[ith_optimal_option_index][track_index]=single_target_hypothesis_index

                '''
                please refer to figure 2 of [1]. The following part discribes track 3 which is a newly initiated track based on measurements of this frame. 
                Notice there is a discrepensy on the definition of single target hypothesis: A single-target hypothesis corresponds to a
                sequence of measurements associated to a potentially detected target. 
                This step is also de facto birth of a new track because now this track would be part of the global hypothesis
                for the next step prediction and update step.             
                
                For newly established potential tracks, there are only two hypothesis
                h_-1 which means this track does not exist
                h_0 which means this track exist the indexing of the track is n_previous measurement + measurement index and the associated measurement is measurement_index. 
                '''
                for i in range(number_of_measurements_from_current_frame):
                    potential_new_track_index = number_of_detected_targets_from_previous_frame + i
                    for ith_optimal_option_index, ith_optimal_option_vector in enumerate(optimal_associations_all): # get he number of row vectors of opt_indices_trans
                        indices_of_ith_optimal_option_associated_measurement_list = [idx for idx, value in enumerate(ith_optimal_option_vector) if value == potential_new_track_index] # if this track is part of optimal single target hypothesis          
                        if len(indices_of_ith_optimal_option_associated_measurement_list)==0: # if under this global hypothesis, this track is not associated with any measurement, therefore a missed detection
                            #single_target_hypothesis_index=single_target_hypothesis_index_specified_by_previous_step_global_hypothesis # it is a missed detection
                            single_target_hypothesis_index = -1
                            # this is the missed detection hypothesis
                        else:
                            single_target_hypothesis_index = 0
                        globHyp_from_current_frame_under_this_globHyp_from_previous_frame[ith_optimal_option_index][potential_new_track_index]=single_target_hypothesis_index 

                # Each global hypothesis from previous frame will generate k_best_global_hypotheses_under_this_previous_global_hypothesis global hypotheses
                # After looping over this global hypothesis of previous frame and generating corresponding data structure

                for ith_optimal_global_hypothesis in globHyp_from_current_frame_under_this_globHyp_from_previous_frame:
                    globHyp.append(ith_optimal_global_hypothesis)
            filter_updated['globHyp']=globHyp
            if len(weight_of_global_hypothesis_in_log_format)>0:
                maximum_weight_of_global_hypothesis_in_log_format = np.max(weight_of_global_hypothesis_in_log_format)              
                # Because the previous steps are computed with weight_of_global_hypothesis_in_log_format
                # We need to np.exp in order to convert it back to decimal system.
                # Here every element of weight_of_global_hypothesis_in_log_format is divided by maximum_weight_of_global_hypothesis_in_log_format
                globWeight=[np.exp(x-maximum_weight_of_global_hypothesis_in_log_format) for x in weight_of_global_hypothesis_in_log_format]
                #globWeight = [np.exp(x) for x in weight_of_global_hypothesis_in_log_format]
                # Normalisation of weights of global hypotheses
                globWeight=globWeight/sum(globWeight)
            else:
                globWeight = []
            filter_updated['globHypWeight']=globWeight  

        return filter_updated

    """
    Step 3: State Estimation. Section VI of [1]
    Firstly, obtain the only global hypothesis with the "maximum weight" from remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm). 
    Then the state extraction is based on this only global hypothesis. Sepecifically, there are three ways to obtain this only global hypothesis:
    Option 1. The only global hypothesis is obtained via maximum globHypWeight: maxmum_global_hypothesis_index = argmax(globHypWeight).
    Option 2. First, compute for cardinality. Then compute weight_new according to cardinality. Finally, obtain the maximum only global hypothesis via this new weight via argmax(weight_new).
    Option 3. Generate deterministic cardinality via a fixed eB threshold. Then compute weight_new and argmax(weight_new) the same way as does Option 2.  
    """
    def extractStates(self, filter_updated):
        state_extraction_option = self.model['state_extraction_option']
        # Get data
        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypotheses_at_current_frame = len(globHypWeight)
        number_of_tracks_at_current_frame=len(filter_updated['tracks'])

        # Initiate datastructure
        state_estimate = {}
        mean = []
        covariance = []
        existence_probability = []
        association_history = []

        if state_extraction_option == 1:
            if number_of_global_hypotheses_at_current_frame>0: # If there are valid global hypotheses
                highest_weight_global_hypothesis_index = np.argmax(globHypWeight) # get he index of global hypothesis with largest weight
                highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index] # get the global hypothesis with largest weight
                for track_index in range(len(highest_weight_global_hypothesis)): # number of tracks.
                    single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index]) # Get the single target hypothesis index.
                    if single_target_hypothesis_specified_by_global_hypothesis > -1: # If the single target hypothesis is not does not exist
                        #a =filter_updated['tracks'][track_index]['eB'] 
                        eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                        if eB >self.model['eB_estimation_threshold']: # if the existence probability is greater than the threshold
                            mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis]) # then assume that this track exist.
                            covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_specified_by_global_hypothesis])
                            existence_probability.append(filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis])
                            association_history.append(filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_specified_by_global_hypothesis])
        elif state_extraction_option ==2:
            '''
            This option is discribed by section VI B.
            cardinality is computed first, and then with this cardinality, the global hypothesis weight is recomputed.
            Advantage: More accurate. with cardinality information, you can easily incorporate pertinent pieces from all global hypotheses.
            Disadvantage: more computation.
            '''
            if number_of_global_hypotheses_at_current_frame>0: 
                predicted_cardinality_tot=np.zeros(number_of_tracks_at_current_frame+1) # need to plus one to account the condition where cardinality is 0
                eB_tot=[] # initiate the total existence probability as an empty
                for global_hypothesis_index in range(number_of_global_hypotheses_at_current_frame):
                    eB_for_this_global_hypothesis=np.zeros(number_of_tracks_at_current_frame)
                    for track_index in range(number_of_tracks_at_current_frame):
                        single_target_hypothesis_index=globHyp[global_hypothesis_index][track_index]
                        if single_target_hypothesis_index != -1: # If this track exist
                            eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_index]
                            eB_for_this_global_hypothesis[track_index]=eB
        
                    eB_tot.append(eB_for_this_global_hypothesis)
                    # predicted cardinality is computed in accordance with equation 48 of [1]
                    predicated_cardinality_for_this_global_hypothesis=CardinalityMB(eB_for_this_global_hypothesis) 
                    # Multiply by weights
                    predicted_cardinality_tot+=globHypWeight[global_hypothesis_index]*predicated_cardinality_for_this_global_hypothesis
                
                #We estimate the cardinality
                max_cardinality_index=np.argmax(predicted_cardinality_tot)
                card_estimate=max_cardinality_index
    
                if card_estimate>0: # If there are tracks for this frame.
                    # Now we go through all hypotheses again
                    weight_hyp_card=[int(x) for x in np.ones(number_of_global_hypotheses_at_current_frame)]
                    indices_sort_hyp=[]
               
                    for global_hypothesis in range(number_of_global_hypotheses_at_current_frame):
                        weight_for_each_single_target_hypothesis_under_this_global_hypothesis=eB_tot[global_hypothesis]
                        sorted_list = [[value,idx] for idx, value in enumerate(weight_for_each_single_target_hypothesis_under_this_global_hypothesis.sort(reverse=True))]
                        sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis = [sorted_list[x][0] for x in range(len(weight_for_each_single_target_hypothesis_under_this_global_hypothesis))]
                        sorted_index_for_each_single_target_hypothesis_under_this_global_hypothesis = [sorted_list[x][1] for x in range(len(weight_for_each_single_target_hypothesis_under_this_global_hypothesis))]

                        indices_sort_hyp.append(sorted_index_for_each_single_target_hypothesis_under_this_global_hypothesis)
                        # the following compuation is done in accordance with equation 49 of [1]
                        vector=sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis[:card_estimate+1]+(1-sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis[card_estimate+1:])
                        # weight_new = sum over cardinality existing tracks's weight*eB + sum over non-existence track's weight*(1-eB)
                        weight_hyp_card[global_hypothesis]*=reduce(operator.mul, vector, 1) # Equation 49 of [1]
                
                    maximum_global_hypothesis_index = np.argmax(weight_hyp_card) # Get the index of the highest weight global hypothesis
                    sorted_index_for_single_target_hypothesis_of_optimal_global_hypothesis=indices_sort_hyp[maximum_global_hypothesis_index]
                    global_hypothesis=globHyp[maximum_global_hypothesis_index]
                    # Gasia's way of initiating the states NOT SURE WHY HE DID IT THIS WAY

                    for track_index in card_estimate:
                        target_i=sorted_index_for_single_target_hypothesis_of_optimal_global_hypothesis[track_index] # Target Index
                        hyp_i=global_hypothesis[target_i]
                        # Gasia's way of indexing this 
                        mean.append(filter_updated['tracks'][target_i]['meanB'][hyp_i])
                        covariance.append(filter_updated['tracks'][target_i]['covB'][hyp_i])
                        existence_probability.append(filter_updated['tracks'][target_i]['eB'][hyp_i])
                        association_history.append(filter_updated['tracks'][target_i]['measurement_association_history'][hyp_i])
                        
        elif state_extraction_option==3:
            '''
            This option is discribed by section VI C.
            Instead of compute for an accurate cardinality, the cardinality is estimated by setting eB == 0.5. 
            If the eB is greater than 0.5, we assume it exist with this probability and the multiply it with weight.
            Otherwise, we assume this track does not exist, with probability (1-eB) and multiply it with weight
            and then with this newly computed weight for global hypothesis, we choose the one with maximum weight.
            Advantage: relatively accurate. can incorporate inforamtion from more than one global hypothesis.
            '''
            if number_of_global_hypotheses_at_current_frame>0:
                new_weights=copy.deepcopy(globHypWeight)
                for global_hypothesis_index in range(number_of_global_hypotheses_at_current_frame):
                    for track_index in range(number_of_tracks_at_current_frame):
                        single_target_hypothesis_index=globHyp[global_hypothesis_index][track_index]
                        if single_target_hypothesis_index != -1: # If this track exist
                            eB=filter_updated['tracks'][track_index]['eB'][int(single_target_hypothesis_index)]
                            # the following is computed in accordance with equation 50 of [1]
                            # this is actually a way to hard code cardinality
                            # instead of compute the cardinality distribution as does in option 2, just set a random value to determine if this track exist
                            if eB >0.5: 
                                new_weights[global_hypothesis_index]*=eB # if this track exist, weight * eB
                            else:
                                new_weights[global_hypothesis_index]*=(1-eB) # if this track does not exist, weigt * (1-eB)
            
                maximum_weight_global_hypothesis_index=np.argmax(new_weights)   # select the hightest weight global hypothesis
                
                # Extract states based on the new weights
                for track_index in range(number_of_tracks_at_current_frame):
                    single_target_hypothesis_index=int(globHyp[maximum_weight_global_hypothesis_index][track_index])
                    if single_target_hypothesis_index>-1: # If this track exist
                        eB=filter_updated['tracks'][track_index]['eB'][int(single_target_hypothesis_index)]
                        if eB>0.5:
                            mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_index])
                            covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_index])
                            existence_probability.append(filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_index])
                            association_history.append(filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_index])

        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['existence_probability'] = existence_probability
        state_estimate['measurement_association_history'] = association_history

        return state_estimate
    
    """
    Step 4: Pruning
    4.1. Prune the Poisson part by discarding components whose weight is below a threshold.
    4.2. Prune the global hypothese by discarding components whose weight is below a threshold.
    4.3. Prune Multi-Bernoulli RFS:
    4.3.1. Mark single target hypothese whose existence probability is below a threshold.
    4.3.2. Based on the marks of previous step, delete tracks whose single target hypothesis are all below the threshold. 
    4.3.2. Remove Bernoulli components do not appear in the remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm).
            
    By doing this, only the single target hypotheses belong to the k best global hypotheses will be left, propogated to next frame as "root" to generate more
    single target hypotheses at next frame.
    In other words, more than one global hypotheses(i.e. the k best global hypothese) will be propagated into next frame as "base" to generate 
    more global hypotheses for next frame. This is why people claim that PMBM is a MHT like filter(in MHT, the multiple hypotheses are propogated 
    from previous frame to current frame thus generating more hypotheses based the "base" multiple hypotheses from previous frame, and the best 
    hypothesis is selected (like GNN) among all the generated hypotheses at current frame.
    """ 
    
    def prune(self, filter_updated):
        # initiate filter_pruned as a copy of filter_updated
        filter_pruned = copy.deepcopy(filter_updated)

        # extract pertinent data from the dictionary
        weightPois=copy.deepcopy(filter_updated['weightPois'])
        global_hypothesis_weights=copy.deepcopy(filter_updated['globHypWeight'])
        globHyp=copy.deepcopy(filter_updated['globHyp'])
        maximum_number_of_global_hypotheses = self.model['maximum_number_of_global_hypotheses']
        eB_threshold = self.model['eB_threshold']
        Poisson_threshold = self.model['T_pruning_Pois']
        MBM_threshold = self.model['T_pruning_MBM']
        """
        Step 4.1.
        Prune the Poisson part by discarding components whose weight is below a threshold.
        """
        # if weight is smaller than the threshold, remove the Poisson component
        indices_to_remove_poisson=[index for index, value in enumerate(weightPois) if value<Poisson_threshold]
        for offset, idx in enumerate(indices_to_remove_poisson):
            del filter_pruned['weightPois'][idx-offset]
            del filter_pruned['meanPois'][idx-offset]
            del filter_pruned['covPois'][idx-offset]
        """
        Step 4.2.
        Pruning Global Hypothesis
        Any global hypothese with weights smaller than the threshold would be pruned away. 
        """
        # only keep global hypothesis whose weight is larger than the threshold
        indices_to_keep_global_hypotheses=[index for index, value in enumerate(global_hypothesis_weights) if value>MBM_threshold]
        weights_after_pruning_before_capping=[global_hypothesis_weights[x] for x in indices_to_keep_global_hypotheses]
        globHyp_after_pruning_before_capping=[globHyp[x] for x in indices_to_keep_global_hypotheses]
        # negate the list first because we require the descending order instead of acending order.
        weight_after_pruning_negative_value = [-x for x in weights_after_pruning_before_capping]
        # If after previous step there are still more global hypothesis than desirable:
        # Pruning components so that there is at most a maximum number of components.
        # get the indices for global hypotheses in descending order.
        index_of_ranked_global_hypothesis_weights_in_descending_order=np.argsort(weight_after_pruning_negative_value) # the index of elements in ascending order
        if len(weights_after_pruning_before_capping)>maximum_number_of_global_hypotheses:
            # cap the list with maximum_number_of_global_hypotheses
            indices_to_keep_global_hypotheses_capped = index_of_ranked_global_hypothesis_weights_in_descending_order[:maximum_number_of_global_hypotheses]
        else:
            indices_to_keep_global_hypotheses_capped=index_of_ranked_global_hypothesis_weights_in_descending_order[:len((weights_after_pruning_before_capping))]
        
        
        globHyp_after_pruning = [copy.deepcopy(globHyp_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped] 
        weights_after_pruning = [copy.deepcopy(weights_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped]
        # normalize weight of global hypotheses
        weights_after_pruning=[x/np.sum(weights_after_pruning) for x in weights_after_pruning]
        globHyp_after_pruning=np.array(globHyp_after_pruning)
        weights_after_pruning=np.array(weights_after_pruning)

        """
        Step 4.3.1.
        Mark Bernoulli components whose existence probability is below a threshold
        Notice it is just to mark those components with existance probability that is below the threshold.
        We don't just simply remove those tracks, since even for elements with small existance probability,
        it is still possible that it is part of the k most optimal global hypothese.

        Notice that this is the only path that would lead to a path deletion.
        """
        
        for track_index in range(len(filter_pruned['tracks'])):     
            # Get the indices for single target hypotheses that is lower than the threshold.
            indices_of_single_target_hypotheses_to_be_marked=[index for index,value in enumerate(filter_pruned['tracks'][track_index]['eB']) if value < eB_threshold]
            # If there is a single target hypothesis that should be removed but is part of global hypotheses
            #global_hypothesis_for_this_track_single_target_hypothesis_to_be_marked = []
            for single_target_hypothesis_to_be_marked_idx in indices_of_single_target_hypotheses_to_be_marked:
                # check each element of the single target hypothese made up for the global hypotheses
                for index_of_single_target_hypothesis_in_global_hypothesis,single_target_hypothesis_in_global_hypothesis in enumerate(globHyp_after_pruning[:,track_index]):
                    # if this single target hypothesis that is below the threshold is present at this global hypothesis
                    if single_target_hypothesis_in_global_hypothesis==single_target_hypothesis_to_be_marked_idx:
                        # mark it with -1, which would be utilized in the next step to initate track deletion.
                        globHyp_after_pruning[:,track_index][index_of_single_target_hypothesis_in_global_hypothesis]=-1
        # if the column vector sums to 0 means this track does not participate in any global hypothesis
        """
        Step 4.3.2.
        Remove tracks(Bernoulli components) that do not take part in any global hypothesis 
        """  
        # check if all the single target hypothesis under this track that participates in the global hypothesis
        # has existence probability that is below the threhold
        tracks_to_be_removed = [x for x in range(len(globHyp_after_pruning[0])) if np.sum(globHyp_after_pruning[:,x]) == -len(globHyp_after_pruning)]
        if len(tracks_to_be_removed)>0:
            #filter_pruned['tracks'] = np.array(filter_pruned['tracks'],dtype=object)
            #filter_pruned['tracks'] = np.delete(filter_pruned['tracks'],tracks_to_be_removed)
            #filter_pruned['tracks'] = list(filter_pruned['tracks'])
            # the np.delete method above does not work
            # empirically, del of a list element is the more stable of the two here
            # delete the tracks whose single target hypothese has existence probability below threshold
            for offset, track_index_to_be_removed in enumerate(tracks_to_be_removed):
                del filter_pruned['tracks'][track_index_to_be_removed-offset]
            # after the track deletion, delete the corresponding column vectors from global hypothesis matrix.
            globHyp_after_pruning = np.delete(globHyp_after_pruning, tracks_to_be_removed, axis=1)
        for track_index in range(len(filter_pruned['tracks'])): # notice that the number of tracks has changed
            """
            Step 4.3.3.
            Remove single-target hypotheses in each track (Bernoulli component) that do not belong to any global hypothesis.
            """  
            single_target_hypothesis_indices_to_be_removed = []            
            number_of_single_target_hypothesis =len(filter_pruned['tracks'][track_index]['eB'])
            
            # read out the single target hypothese participate in the global hypotheses
            valid_single_target_hypothesis_for_this_track = globHyp_after_pruning[:,track_index]
            # if a single target hypothesis does not participate in any global hypothesis
            # then it would be removed
           
            for single_target_hypothesis_index in range(number_of_single_target_hypothesis):
                # if this single target hypothesis does not particulate in the global hypotheses
                if single_target_hypothesis_index not in valid_single_target_hypothesis_for_this_track:
                    # add it to the to be removed list.
                    single_target_hypothesis_indices_to_be_removed.append(single_target_hypothesis_index)
            # if there are single target hypothese to be removed
            if len(single_target_hypothesis_indices_to_be_removed)>0:
                # remove the single target hypotheses from the data structure
                for offset, index in enumerate(single_target_hypothesis_indices_to_be_removed):
                    del filter_pruned['tracks'][track_index]['eB'][index-offset]
                    del filter_pruned['tracks'][track_index]['meanB'][index-offset]
                    del filter_pruned['tracks'][track_index]['covB'][index-offset]
                    del filter_pruned['tracks'][track_index]['weight_of_single_target_hypothesis_in_log_format'][index-offset]
                    del filter_pruned['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame'][index-offset]
                    del filter_pruned['tracks'][track_index]['measurement_association_history'][index-offset]
                    del filter_pruned['tracks'][track_index]['measurement_association_from_this_frame'][index-offset]
            """
            Step 4.3.4.
            Adjust the global hypothesis indexing if there are deletions of single target hypotheses. 
            For instance, in the global hypothesis, the current single target hypothesis is indexed as 5.
            But during previous steps, single target hypothesis 3, 6 is pruned away,
            therefore, the remaining single target hypothesis number 5 need to be adjusted to 4. 
            """ 
            # after the deletion, readjust the indexing system
            if len(single_target_hypothesis_indices_to_be_removed)>0:
                for global_hypothesis_index, global_hypothesis_vector in enumerate(globHyp_after_pruning):
                    # find out how many single target hypothesis are deleted before this single target hypothesis
                    single_target_hypothesis_specified_by_the_global_hypothesis = global_hypothesis_vector[track_index]
                    # the index of removed single target hypothesis before the end point
                    single_target_hypotheses_removed_before_this_single_taget_hypothesis = [x for x in single_target_hypothesis_indices_to_be_removed if x<single_target_hypothesis_specified_by_the_global_hypothesis]
                    # adjust the indexing system by the length of removed terms before this element.
                    subtraction=len(single_target_hypotheses_removed_before_this_single_taget_hypothesis)
                    globHyp_after_pruning[global_hypothesis_index][track_index]-=subtraction
  
        """
        Step 4.3.5.
        Merge the duplicated global hypothese if there are any. 
        """ 
        # After the previous steps of pruning, 
        # there can be duplication amongst the remaining global hypotheses. 
        # so the solution is to merged those duplications by adding their weights 
        # and leave only one element. 

        # get the unique elements
        globHyp_unique, indices= np.unique(globHyp_after_pruning, axis=0, return_index = True)
        # get the indices of duplication
        duplicated_indices = [x for x in range(len(globHyp_after_pruning)) if x not in indices]
        # check if there are any duplication.
        if len(globHyp_unique)!=len(globHyp_after_pruning): #There are duplicate entries
            weights_unique=np.zeros(len(globHyp_unique))
            for i in range(len(globHyp_unique)):
                # fill in the weight of unique elements
                weights_unique[i] = global_hypothesis_weights[indices[i]]
                for j in duplicated_indices:
                    if list(globHyp_after_pruning[j]) == list(globHyp_unique[i]):
                        # add the weight of duplications to its respective unique elements.
                        weights_unique[i]+=global_hypothesis_weights[j]

            globHyp_after_pruning=globHyp_unique
            weights_after_pruning=weights_unique
            weights_after_pruning/sum(weights_after_pruning)
    
        filter_pruned['globHyp']=globHyp_after_pruning
        filter_pruned['globHypWeight']=weights_after_pruning
        return filter_pruned