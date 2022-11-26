from cv2 import normalize
import numpy as np
import copy
import math
from trackers.PMBMGNN.murty import Murty
from trackers.PMBMGNN.util import mvnpdf, CardinalityMB
from functools import reduce
import operator
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis as mah
from utils.utils import associate_dets_to_tracks
from utils.utils import giou3d, giou2d,readout_parameters
import json

# read parameters
with open("/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/configs/pmbmgnn_parameters.json", 'r') as f:
    parameters=json.load(f)

class PMBMGNN_Filter:

    def __init__(self, model): 
        self.model = model # use generated model which is configured for all parameters used in PMBM filter model for tracking the multi-targets.

    def predict(self,egoposition,lag_time, filter_pruned,Z_k, birth_rate, noisy_region=1):

  
        F = np.eye(4, dtype=np.float64)
        I = lag_time*np.eye(2, dtype=np.float64)
        F[0:2, 2:4] = I
        Q = self.model['Q_k']   # Process noise, Q.
        #number_of_new_birth_targets = self.model['number_of_new_birth_targets']
        number_of_surviving_previously_detected_targets=len(filter_pruned['tracks'])
        
        filter_predicted = {}

        # MBM Components data structure
        if number_of_surviving_previously_detected_targets > 0:
            filter_predicted['tracks'] = [{} for i in range(number_of_surviving_previously_detected_targets)]
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = copy.deepcopy(filter_pruned['globHyp'])
            filter_predicted['globHypWeight'] = copy.deepcopy(filter_pruned['globHypWeight'])
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                filter_predicted['tracks'][previously_detected_target_index]['eB']=[] # need to be filled in with prediction value                
                filter_predicted['tracks'][previously_detected_target_index]['meanB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['covB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['rotationB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['elevationB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['classificationB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['idB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB']=[]
                #filter_predicted['tracks'][previously_detected_target_index]['max_idB']=filter_pruned['tracks'][previously_detected_target_index]['max_idB']  
                filter_predicted['tracks'][previously_detected_target_index]['sizeB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'])
                filter_predicted['tracks'][previously_detected_target_index]['giou']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['giou'])
                filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['measurement_association_history'])
                #filter_predicted['tracks'][previously_detected_target_index]['track_establishing_frame']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['track_establishing_frame'])
                filter_predicted['tracks'][previously_detected_target_index]['association_counter']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['association_counter'])

        else:
            filter_predicted['tracks'] = []
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = []
            filter_predicted['globHypWeight'] = []

        """
        Step 1.3 : Prediction for existing/surviving previously detected targets(i.e. targets were detected at previous frame and survive into current frame) by using Bernoulli components, or so called Multi-Bernoulli RFS.
        """
        if number_of_surviving_previously_detected_targets > 0:
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                for single_target_hypothesis_index_from_previous_frame in range(len(filter_pruned['tracks'][previously_detected_target_index]['eB'])):
                    # Get data from previous frame
                    eB_previous = filter_pruned['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                    meanB_previous = filter_pruned['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                    detection_scoreB_previous = filter_pruned['tracks'][previously_detected_target_index]['detection_scoreB'][single_target_hypothesis_index_from_previous_frame]
                    covB_previous = filter_pruned['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                    rotationB_previous = filter_pruned['tracks'][previously_detected_target_index]['rotationB'][single_target_hypothesis_index_from_previous_frame]       
                    elevationB_previous = filter_pruned['tracks'][previously_detected_target_index]['elevationB'][single_target_hypothesis_index_from_previous_frame]       
                    sizeB_previous = filter_pruned['tracks'][previously_detected_target_index]['sizeB'][single_target_hypothesis_index_from_previous_frame]       
                    classificationB_previous = filter_pruned['tracks'][previously_detected_target_index]['classificationB'][single_target_hypothesis_index_from_previous_frame]
                    birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classificationB_previous, parameters)
                    idB_previous = filter_pruned['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                    
                    #eB_predicted = detection_scoreB_previous * eB_previous
                    Ps = P_s
                    meanB_predicted = F.dot(meanB_previous)
                    distance=math.sqrt((egoposition[0]-meanB_predicted[0][0])**2+(egoposition[0]-meanB_predicted[0][0])**2)
                    #if distance > 20:
                    #    Ps*=0.5
                    if distance > 60:
                        Ps=0.1
                    eB_predicted = Ps * eB_previous

                    covB_predicted = F.dot(covB_previous).dot(np.transpose(F)) + Q
                    # Fill in the data structure                   
                    filter_predicted['tracks'][previously_detected_target_index]['eB'].append(eB_predicted)                    
                    filter_predicted['tracks'][previously_detected_target_index]['meanB'].append(meanB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['covB'].append(covB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['idB'].append(idB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB'].append(detection_scoreB_previous)
                    
        return filter_predicted

    def predict_initial_step(self, Z_k, birth_rate,noisy_region=1):
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
        filter_predicted['detection_scorePois']=[]
        filter_predicted['rotationPois']=[]
        filter_predicted['elevationPois']=[]
        filter_predicted['classificationPois']=[]
        filter_predicted['sizePois']=[]
        filter_predicted['tracks'] = []
        filter_predicted['max_idB']=0
        filter_predicted['globHyp'] = []
        filter_predicted['globHypWeight'] = []
        filter_predicted['idPois']=[]
        filter_predicted['max_idPois']=len(Z_k)
        # Get the parameters
        number_of_new_birth_targets_init = len(Z_k)

        trans_width = noisy_region

        for new_birth_target_index in range(len(Z_k)):
            delta_x = np.random.uniform(-trans_width, trans_width)
            delta_y = np.random.uniform(-trans_width, trans_width)

            # Compute for the birth initiation
            weightPois_birth = birth_rate
            meanPois_birth=np.array([delta_x+Z_k[new_birth_target_index]['translation'][0],delta_y+Z_k[new_birth_target_index]['translation'][1],Z_k[new_birth_target_index]['velocity'][0],Z_k[new_birth_target_index]['velocity'][1]]).reshape(-1,1).astype(np.float64)
            #meanPois_birth=np.array([delta_x+Z_k[new_birth_target_index]['translation'][0],delta_y+Z_k[new_birth_target_index]['translation'][1],0,0]).reshape(-1,1).astype(np.float64)

            covPois_birth = self.model['P_new_birth']
            # Fill lin the data structure
            filter_predicted['weightPois'].append(weightPois_birth)  # Create the weight of PPP using the weight of the new birth PPP
            #filter_predicted['weightPois'].append(Z_k[new_birth_target_index]['detection_score']*weightPois_birth)
            filter_predicted['meanPois'].append(meanPois_birth)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(covPois_birth)    # Create the variance of PPP using the variance of the new birth PPP
            filter_predicted['rotationPois'].append(Z_k[new_birth_target_index]['rotation'])
            filter_predicted['elevationPois'].append(Z_k[new_birth_target_index]['translation'][2])
            filter_predicted['classificationPois'].append(Z_k[new_birth_target_index]['detection_name'])
            filter_predicted['sizePois'].append(Z_k[new_birth_target_index]['size'])
            filter_predicted['detection_scorePois'].append(Z_k[new_birth_target_index]['detection_score'])
            filter_predicted['idPois'].append(new_birth_target_index)
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
    def update(self, Z_k, filter_predicted, confidence_score=0, giou_gating=0.15):
        # Get pre-defined parameters.
        H = self.model['H_k'] # measurement model
        R = self.model['R_k'] # measurement noise
        Pd =self.model['p_D'] # probability for detection


        po_gating_threshold = self.model['poission_gating_threshold']
        ber_gating_threshold = self.model['bernoulli_gating_threshold']
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
        filter_updated['detection_scorePois'] = []
        filter_updated['rotationPois']=[]
        filter_updated['elevationPois']=[]
        filter_updated['classificationPois']=[]
        filter_updated['sizePois']=[]
        filter_updated['idPois']=[]
        filter_updated['max_idPois']=filter_predicted['max_idPois']

        if number_of_detected_targets_from_previous_frame==0:
            # Initiate the globHyp as all zeros which means newly generated track associate with itself
            filter_updated['globHyp']=[[int(x) for x in np.zeros(number_of_measurements_from_current_frame)]] # The global hypothesis is all associated with missed detection
            filter_updated['globHypWeight']=[1] # There would be one global hypothesis, each measurement is associated with itself.
            if number_of_measurements_from_current_frame == 0:
                filter_updated['tracks'] = []
                filter_updated['max_idB']=0            
            else: 
                filter_updated['tracks']=[{} for n in range(number_of_measurements_from_current_frame)] # Initiate the data structure with right size of dictionaries
                for i in range(number_of_measurements_from_current_frame): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][i]['eB']= []
                    filter_updated['tracks'][i]['covB']= []
                    filter_updated['tracks'][i]['meanB']= []
                    filter_updated['tracks'][i]['rotationB']=[]
                    filter_updated['tracks'][i]['elevationB']=[]
                    filter_updated['tracks'][i]['classificationB']=[]
                    filter_updated['tracks'][i]['idB']=[]
                    filter_updated['tracks'][i]['detection_scoreB']=[]
                    filter_updated['tracks'][i]['sizeB']=[]
                    filter_updated['tracks'][i]['weight_of_single_target_hypothesis_in_log_format']= []
                    filter_updated['tracks'][i]['giou']= []
                    filter_updated['tracks'][i]['single_target_hypothesis_index_from_previous_frame']=[]
                    filter_updated['tracks'][i]['measurement_association_history']= []
                    filter_updated['tracks'][i]['measurement_association_from_this_frame']= []
                    filter_updated['tracks'][i]['association_counter']= []
            filter_updated['max_idB']=filter_predicted['max_idB']+number_of_measurements_from_current_frame
        else:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            filter_updated['tracks']=[{} for n in range(number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame)] # Initiate the data structure with right size of dictionaries
            # Initiate data structure for indexing 0 to number of detected target index
            for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
                filter_updated['tracks'][previously_detected_target_index]['eB'] = []
                filter_updated['tracks'][previously_detected_target_index]['meanB'] = []
                filter_updated['tracks'][previously_detected_target_index]['covB'] = []
                filter_updated['tracks'][previously_detected_target_index]['idB'] = []
                filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'] = []
                #filter_updated['tracks'][previously_detected_target_index]['max_idB'] = filter_predicted['tracks'][previously_detected_target_index]['max_idB']+number_of_measurements_from_current_frame+1
                filter_updated['tracks'][previously_detected_target_index]['rotationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['elevationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['classificationB'] = []
                filter_updated['tracks'][previously_detected_target_index]['sizeB'] = []
                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'] = []
                filter_updated['tracks'][previously_detected_target_index]['giou'] = []
                filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame']=[]
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'] = copy.deepcopy(filter_predicted['tracks'][previously_detected_target_index]['measurement_association_history'])
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'] = []
                filter_updated['tracks'][previously_detected_target_index]['association_counter'] = []


            # Initializing data structure for index from number of previously detected targets to number of previosly detected targets + number of measuremetns                  
            for i in range(number_of_measurements_from_current_frame): # Initialte the dictionary with empty list.
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['eB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['meanB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['covB']= []
                #filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['max_idB'] = filter_predicted['tracks'][number_of_detected_targets_from_previous_frame]['max_idB']+number_of_measurements_from_current_frame+1
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['idB']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['detection_scoreB']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['rotationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['elevationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['classificationB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['sizeB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['weight_of_single_target_hypothesis_in_log_format']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['giou']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['single_target_hypothesis_index_from_previous_frame']=[]
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_history']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['measurement_association_from_this_frame']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['association_counter']= []

            filter_updated['max_idB']=filter_predicted['max_idB']+number_of_measurements_from_current_frame
  
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
            detection_scorePois_predicted = filter_predicted['detection_scorePois'][PPP_component_index]
            rotationPois_predicted = filter_predicted['rotationPois'][PPP_component_index]
            elevationPois_predicted = filter_predicted['elevationPois'][PPP_component_index]
            classificationPois_predicted = filter_predicted['classificationPois'][PPP_component_index]
            sizePois_predicted = filter_predicted['sizePois'][PPP_component_index]
            idPois_predicted = filter_predicted['idPois'][PPP_component_index]

            # Compute for update
            if self.model['use_ds_for_pd']:
                wegithPois_updated = (1-filter_predicted['detection_scorePois'][PPP_component_index]) * weightPois_predicted
            else:
                wegithPois_updated = (1-Pd) * weightPois_predicted
            
            meanPois_updated = meanPois_predicted # because it is miss detected, therefore, no updating required
            covPois_updated = covPois_predicted # ditto
            
            # Fill in the data structure
            filter_updated['weightPois'].append(wegithPois_updated)
            filter_updated['meanPois'].append(meanPois_updated)
            filter_updated['covPois'].append(covPois_updated)
            filter_updated['rotationPois'].append(rotationPois_predicted)
            filter_updated['elevationPois'].append(elevationPois_predicted)
            filter_updated['detection_scorePois'].append(detection_scorePois_predicted)
            filter_updated['classificationPois'].append(classificationPois_predicted)
            filter_updated['sizePois'].append(sizePois_predicted)
            filter_updated['idPois'].append(idPois_predicted)
        filter_updated['max_idPois']=filter_predicted['max_idPois']

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
                
                ppp_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_PPP_component_measured[0],Z_k[measurement_index]['translation'][1] - mean_PPP_component_measured[1]]).reshape(-1,1).astype(np.float64)
                Si = copy.deepcopy(S_PPP_component)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64)) #dim 2x2
                
                track={}
                temp=mean_PPP_component_predicted.reshape(-1,1).tolist()
                track['translation']=[temp[0][0],temp[1][0]]
                track['translation'].append(elevationPois_predicted)
                track['rotation']=rotationPois_predicted
                track['size']=sizePois_predicted

                if self.model['gating_mode']=='mahalanobis':
                    maha1 = np.transpose(ppp_innovation_residual).dot(invSi).dot(ppp_innovation_residual)[0][0]
                    value=maha1
                    gating_threshold=ber_gating_threshold
                    if value < gating_threshold: 
                        tracks_associated_with_this_measurement.append(PPP_component_index)                
                elif self.model['gating_mode']=='giou':
                    value = giou3d(Z_k[measurement_index],track)
                    gating_threshold=giou_gating
                    if value >= gating_threshold: 
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
                    track_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_associated_track_measured[0],Z_k[measurement_index]['translation'][1] - mean_associated_track_measured[1]]).reshape(-1,1).astype(np.float64)
                    
                    # Compute for update
                    mean_associated_track_updated = mean_associated_track_predicted + K_associated_track.dot(track_innovation_residual) # it is a column vector with lenghth 4
                    #mean_associated_track_updated[2][0]=Z_k[measurement_index]['velocity'][0]
                    #mean_associated_track_updated[3][0]=Z_k[measurement_index]['velocity'][1]

                    cov_associated_track_updated = cov_associated_track_predicted - K_associated_track.dot(H).dot(cov_associated_track_predicted).astype('float64')
                    # For numerical stability
                    cov_associated_track_updated = 0.5 * (cov_associated_track_updated + np.transpose(cov_associated_track_updated))               
                    mvnpdf_value=mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_associated_track_measured[0],mean_associated_track_measured[1]]),S_associated_track)
                    if self.model['use_ds_for_pd']:
                        weight_for_track_detection = Z_k[measurement_index]['detection_score']*weight_associated_track_predicted*mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_associated_track_measured[0],mean_associated_track_measured[1]]),S_associated_track)
                    else:
                        weight_for_track_detection = Pd*weight_associated_track_predicted*mvnpdf_value
                    

                    # according to equation 45 of [1]
                    weight_of_true_detection += weight_for_track_detection
                    meanB_sum += weight_for_track_detection*(mean_associated_track_updated)
                    covB_sum += weight_for_track_detection*cov_associated_track_updated + weight_for_track_detection*(mean_associated_track_updated.dot(np.transpose(mean_associated_track_updated)))


                meanB_updated=meanB_sum/weight_of_true_detection
                covB_updated = covB_sum/weight_of_true_detection - (meanB_updated*np.transpose(meanB_updated))
                rotationB_updated = Z_k[measurement_index]['rotation']
                elevationB_updated = Z_k[measurement_index]['translation'][2]
                classificationB_updated = Z_k[measurement_index]['detection_name']
                sizeB_updated = Z_k[measurement_index]['size']

                probability_of_detection = weight_of_true_detection + clutter_intensity
                if Z_k[measurement_index]['detection_score'] > confidence_score:
                    eB_updated = 1
                else:
                    eB_updated = weight_of_true_detection/probability_of_detection
                # Fill in the data structure
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated) # Notice the meaning of of eB, is for cardinality computation, weightB is the hypothesis probability, which are two different things
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['rotationB'].append(rotationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['elevationB'].append(elevationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['classificationB'].append(classificationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['idB'].append(filter_predicted['max_idB']+measurement_index+1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['sizeB'].append(sizeB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection)) # weightB is used for cost matrix computation
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['giou'].append(probability_of_detection) # weightB is used for cost matrix computation

                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(measurement_index) # register history 
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(measurement_index) # register history 
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['association_counter'].append(1) # register history 

            else: 
                '''
                2.2.2
                If there is not any PPP component(previously miss-detected target or new birth target) could be associated with current measurement, assume this measurement is originated from clutter. 
                We still need to create a Bernoulli component for it, since we need to guarantee that ever measurement generate a Bernoulli RFS.
                The created Bernoulli component has existence probability zero (denote it is clutter). It will be removed by pruning.
                '''
                meanB_updated = mean_PPP_component_predicted
                covB_updated = cov_PPP_component_predicted
                rotationB_updated = Z_k[measurement_index]['rotation']
                elevationB_updated = Z_k[measurement_index]['translation'][2]
                sizeB_updated = Z_k[measurement_index]['size']
                classificationB_updated=Z_k[measurement_index]['detection_name']
                #classificationB_updated=classificationB_single_target_hypothesis_predicted

                # weight_of_true_detection = 0 therefore existence probability is 0
                probability_of_detection = clutter_intensity #This measurement is a clutter
                eB_updated = 0

                # in the global hypothesis generating part of the code, this option would be registered as h_-1 this track does not exist
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['eB'].append(eB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['weight_of_single_target_hypothesis_in_log_format'].append(np.log(probability_of_detection))
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['giou'].append(np.log(0.15))

                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['meanB'].append(meanB_updated) # This can be set to zero, essentially, it doesn't matter
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['covB'].append(covB_updated) # ditto
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['rotationB'].append(rotationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['elevationB'].append(elevationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['sizeB'].append(sizeB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['idB'].append(filter_predicted['max_idB']+measurement_index+1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['classificationB'].append(classificationB_updated)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['single_target_hypothesis_index_from_previous_frame'].append(-1) #-1 means this track does not exist
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_history'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(-1)
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + measurement_index]['measurement_association_from_this_frame'].append(0)

        """
        Step 2.3. for update: Section V-C2 of [1]
        Update for targets which got detected at previous frame.
        """
       
        for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
            
            number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
            #filter_updated['tracks'][previously_detected_target_index]['track_establishing_frame'] = [0]
        
            # Loop through all single target hypotheses belong to global hyptheses from previous frame. 
            for single_target_hypothesis_index_from_previous_frame in range(number_of_single_target_hypotheses_from_previous_frame):
                # Get the data from filter_predicted
                mean_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                cov_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                eB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                rotationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['rotationB'][single_target_hypothesis_index_from_previous_frame]
                elevationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['elevationB'][single_target_hypothesis_index_from_previous_frame]
                classificationB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['classificationB'][single_target_hypothesis_index_from_previous_frame]
                sizeB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['sizeB'][single_target_hypothesis_index_from_previous_frame]
                idB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                association_counter_before = filter_predicted['tracks'][previously_detected_target_index]['association_counter'][single_target_hypothesis_index_from_previous_frame]
                #max_idB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['max_idB']
                detection_scoreB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['detection_scoreB'][single_target_hypothesis_index_from_previous_frame]



                birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classificationB_single_target_hypothesis_predicted, parameters)
        
                ber_gating_threshold = bernoulli_gating
                clutter_intensity = clutter_rate
                Pd =P_d
                Ps=P_s



                track={}
                temp=mean_single_target_hypothesis_predicted.reshape(-1,1).tolist()
                track['translation']=[temp[0][0],temp[1][0]]
                track['translation'].append(elevationB_single_target_hypothesis_predicted)
                track['rotation']=rotationB_single_target_hypothesis_predicted
                track['size']=sizeB_single_target_hypothesis_predicted

                """
                Step 2.3.1. for update: Undetected Hypothesis
                Update the targets got detected previously but get undetected at current frame.
                """
                # Compute for Missed detection Hypothesis
                if self.model['use_ds_for_pd']:
                    probability_for_track_exist_but_undetected = eB_single_target_hypothesis_predicted*(1-detection_scoreB_single_target_hypothesis_predicted)
                else:
                    probability_for_track_exist_but_undetected = eB_single_target_hypothesis_predicted*(1-Pd)
                probability_for_track_dose_not_exit = 1-eB_single_target_hypothesis_predicted
                eB_undetected = probability_for_track_exist_but_undetected/(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit) 
                weight_of_single_target_hypothesis_undetected_in_log_format = np.log(probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit) # does not exist plus exist but not detected, how likely is this hypothesis
                mean_single_target_hypothesis_undetected = mean_single_target_hypothesis_predicted
                cov_single_target_hypothesis_undetected = cov_single_target_hypothesis_predicted
                #giou=probability_for_track_exist_but_undetected+probability_for_track_dose_not_exit
                
                # Fill in the data structure
                filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_undetected)
                filter_updated['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['idB'].append(idB_single_target_hypothesis_predicted)
                #filter_updated['tracks'][previously_detected_target_index]['max_idB']=max_idB_single_target_hypothesis_predicted
                filter_updated['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_undetected)
                filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'].append(detection_scoreB_single_target_hypothesis_predicted)
                filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_undetected_in_log_format)
                filter_updated['tracks'][previously_detected_target_index]['giou'].append(0)

                filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(-1) # -1 as the index for missed detection
                filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(-1) # -1 as the index for missed detection
                filter_updated['tracks'][previously_detected_target_index]['association_counter'].append(association_counter_before)
                """
                Step 2.3.2. for update:
                Update the targets got detected previously and still get detected at current frame.
                Beware what we do here is to update all the possible single target hypotheses and corresponding cost value for every single target hypothesis(each 
                target-measurement possible association pair). The single target hypothese which can happen at the same time will form a global hypothesis(joint 
                event), and all the global hypotheses will be formed exhaustively later by using part of "all the possible single target hypotheses". 
                """
                
                # Compute Kalman Filter Elements
                S_single_target_hypothesis = (H.dot(cov_single_target_hypothesis_predicted).dot(np.transpose(H))+R).astype('float64')
                # For numerical stability
                S_single_target_hypothesis = 0.5 * (S_single_target_hypothesis + np.transpose(S_single_target_hypothesis))
                mean_single_target_hypothesis_measured = H.dot(mean_single_target_hypothesis_predicted).astype('float64')
                
                # alternatively, seems that inverse can be computed this way. 
                Si = copy.deepcopy(S_single_target_hypothesis)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))

                # Garsia's method
                # empirically, this is more stable
                #Vs= np.linalg.cholesky(S_single_target_hypothesis) 
                #Vs = np.matrix(Vs)
                #log_det_S_pred_j= 2*np.log(reduce(operator.mul, np.diag(Vs)))
                #Si = copy.deepcopy(Vs)
                #inv_sqrt_Si = np.linalg.inv(np.array(Si, dtype=np.float64))
                #invSi= inv_sqrt_Si*np.transpose(inv_sqrt_Si)

                # Compute for Kalman Gain K: P * H' * Inv(S)
                K_single_target_hypothesis = cov_single_target_hypothesis_predicted.dot(np.transpose(H)).dot(invSi).astype('float64')
                starting_position_idx = len(filter_updated['tracks'][previously_detected_target_index]['meanB'])
                associated_measurement_counter = 0          
                for measurement_index in range(number_of_measurements_from_current_frame): # starting from m_1, since m_0 means missed detection
                    detected_track_innovation_residual = np.array([Z_k[measurement_index]['translation'][0] - mean_single_target_hypothesis_measured[0],Z_k[measurement_index]['translation'][1] - mean_single_target_hypothesis_measured[1]]).reshape(-1,1).astype(np.float64)
                    if self.model['gating_mode']=='mahalanobis':
                        maha2 = np.transpose(detected_track_innovation_residual).dot(invSi).dot(detected_track_innovation_residual)[0][0]
                        value=maha2
                        gating_threshold=ber_gating_threshold
                        if value < gating_threshold:
                            within_gating = True
                        else:
                            within_gating = False
                    elif self.model['gating_mode']=='giou':
                        value = giou3d(Z_k[measurement_index],track)
                        gating_threshold=giou_gating
                        if value >= gating_threshold:
                            within_gating = True
                        else:
                            within_gating = False
                    #maha3=mah(detected_track_innovation_residual, [[0],[0]], invSi)
                    if within_gating: 
                        associated_measurement_counter += 1
                        # Perform Kalman update with this measurement
                        # M+_current = M-_current + K * (Measurement_current - H * M-_curent)
                        mean_single_target_hypothesis_updated = mean_single_target_hypothesis_predicted + K_single_target_hypothesis.dot(detected_track_innovation_residual) # it is a column vector with lenghth 4
                      
                        # P+_current = P-_current - K * H * P-_current
                        cov_single_target_hypothesis_updated = cov_single_target_hypothesis_predicted - K_single_target_hypothesis.dot(H).dot(cov_single_target_hypothesis_predicted).astype('float64')
                        # For numerical stability
                        cov_single_target_hypothesis_updated = 0.5 * (cov_single_target_hypothesis_updated + np.transpose(cov_single_target_hypothesis_updated))
                        mvnpdf_value = mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis)
                        if self.model['use_ds_for_pd']:
                            weight_of_single_target_hypothesis_updated_in_log_format =np.log(Z_k[measurement_index]['detection_score'] * eB_single_target_hypothesis_predicted * mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis))
                        else:
                            weight_of_single_target_hypothesis_updated_in_log_format =np.log(Pd * eB_single_target_hypothesis_predicted * mvnpdf(np.array([[Z_k[measurement_index]['translation'][0]],[Z_k[measurement_index]['translation'][1]]]), np.array([mean_single_target_hypothesis_measured[0],mean_single_target_hypothesis_measured[1]]),S_single_target_hypothesis))

                        if self.model['use_giou']:
                            giou_value = giou3d(Z_k[measurement_index],track) 
                            if giou_value <= 0:
                                print('giou smaller than 0')
                                giou=-50
                            else:
                                giou=np.log(giou_value)
                        else:
                            giou=0
                        
                        eB_single_target_hypothesis_updated = 1
                        rotationB_single_target_hypothesis_updated = Z_k[measurement_index]['rotation']
                        elevationB_single_target_hypothesis_updated = Z_k[measurement_index]['translation'][2]
                        classificationB_single_target_hypothesis_updated = Z_k[measurement_index]['detection_name']
                        sizeB_single_target_hypothesis_updated = Z_k[measurement_index]['size']
                        idB_single_target_hypothesis_updated = filter_predicted['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                        #max_idB_single_target_hypothesis_updated = filter_predicted['tracks'][previously_detected_target_index]['max_idB']
                        # Fill in the data structure
                        filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['rotationB'].append(rotationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['elevationB'].append(elevationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['classificationB'].append(classificationB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['sizeB'].append(sizeB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['idB'].append(idB_single_target_hypothesis_updated)
                        filter_updated['tracks'][previously_detected_target_index]['detection_scoreB'].append(Z_k[measurement_index]['detection_score'])
                        #filter_updated['tracks'][previously_detected_target_index]['max_idB']=max_idB_single_target_hypothesis_updated
                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'].append(weight_of_single_target_hypothesis_updated_in_log_format)
                        filter_updated['tracks'][previously_detected_target_index]['giou'].append(giou)

                        filter_updated['tracks'][previously_detected_target_index]['single_target_hypothesis_index_from_previous_frame'].append(single_target_hypothesis_index_from_previous_frame)
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_history'].append(measurement_index) # Add current measurement to association history
                        filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'].append(measurement_index) # Add current measurement to association history
                        filter_updated['tracks'][previously_detected_target_index]['association_counter'].append(association_counter_before+1) # Add current measurement to association history

        """
        Step 2.4. for update:
        Update Global Hypotheses as described in section V-C3 in [1].
        The objective of global hypotheses is to select k optimal single target hypotheses to propogate towards the next step.
        """
        if number_of_measurements_from_current_frame == 0:
            filter_updated['globHyp']= [[int(x) for x in np.zeros(number_of_detected_targets_from_previous_frame)]]
            filter_updated['globHypWeight']=[1]
        else:
            if number_of_detected_targets_from_previous_frame>0:
    
                weight_of_global_hypothesis_in_log_format=[]
                globHyp=[]
    
                for global_hypothesis_index_from_pevious_frame in range(number_of_global_hypotheses_from_previous_frame):
                    '''
                    Step 2.4.1 Generate Cost Matrix For Each Global Hypothesis Index from previous frame
                    '''
                    cost_matrix_log=-np.inf*np.ones((number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame , number_of_measurements_from_current_frame))
                    weight_for_missed_detection_hypotheses=np.zeros(number_of_detected_targets_from_previous_frame)
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
                            weight_for_missed_detection_hypotheses[previously_detected_target_index]=missed_detection_hypothesis_weight
                            measurement_association_list_generated_under_this_previous_global_hypothesis = [filter_updated['tracks'][previously_detected_target_index]['measurement_association_from_this_frame'][x] for x in new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[1:]]
                            if len(measurement_association_list_generated_under_this_previous_global_hypothesis) >0:
                                for idx, associated_measurement in enumerate(measurement_association_list_generated_under_this_previous_global_hypothesis):
                                    idx_of_current_single_target_hypothesis = new_single_target_hypotheses_indices_generated_under_this_previous_global_hypothesis[idx+1]
                                    if self.model['use_giou']:
                                        cost_matrix_log[previously_detected_target_index][associated_measurement] = \
                                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][idx_of_current_single_target_hypothesis]\
                                        -missed_detection_hypothesis_weight+filter_updated['tracks'][previously_detected_target_index]['giou'][idx_of_current_single_target_hypothesis]
                                    else:
                                        cost_matrix_log[previously_detected_target_index][associated_measurement] = \
                                        filter_updated['tracks'][previously_detected_target_index]['weight_of_single_target_hypothesis_in_log_format'][idx_of_current_single_target_hypothesis]\
                                        -missed_detection_hypothesis_weight
        
                                        
                    '''
                    Step 2.4.1.2 Fill in the cost matrix with regard to the newly initiated tracks
                    '''
                    for measurement_index in range(number_of_measurements_from_current_frame):
                        if self.model['use_giou']:
                            cost_matrix_log[number_of_detected_targets_from_previous_frame+measurement_index][measurement_index]=np.log(0.15)+filter_updated['tracks'][number_of_detected_targets_from_previous_frame+measurement_index]['weight_of_single_target_hypothesis_in_log_format'][0]
                        else:
                            cost_matrix_log[number_of_detected_targets_from_previous_frame+measurement_index][measurement_index]=filter_updated['tracks'][number_of_detected_targets_from_previous_frame+measurement_index]['weight_of_single_target_hypothesis_in_log_format'][0]

                    '''
                    Step 2.4.2 Genereta the K (which varies each frame) optimal option based on cost matrix
                    1. Remove -Inf rows and columns for performing optimal assignment. We take them into account for indexing later. Columns that have only one value different from Inf are not fed into Murty either.
                    2. Use murky to get the kth optimal options
                    3. Add the cost of the misdetection hypothesis to the cost matrix
                    4. Add back the removed infinity options
                    '''
                    indices_of_cost_matrix_with_valid_elements = 1 - np.isinf(cost_matrix_log)
                    indices_of_measurements_non_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])>1]
                    if len(indices_of_measurements_non_exclusive)>0:
                        indices_of_tracks_non_exclusive=[x for x in range(len(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive]))) if sum(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_measurements_non_exclusive])[x]>0)]
                        cost_matrix_log_non_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_non_exclusive]))[indices_of_tracks_non_exclusive]
                    else:
                        indices_of_tracks_non_exclusive = []
                        cost_matrix_log_non_exclusive = []
                    
                    indices_of_measurements_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])==1]
                    if len(indices_of_measurements_exclusive) > 0:
                        indices_of_tracks_exclusive = [np.argmax(indices_of_cost_matrix_with_valid_elements[:,x]) for x in indices_of_measurements_exclusive]
                        cost_matrix_log_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_measurements_exclusive]))[indices_of_tracks_exclusive]
                    else:
                        indices_of_tracks_exclusive = []
                        cost_matrix_log_exclusive = []
                    
                    if len(cost_matrix_log_non_exclusive)==0:
                        association_vector=np.zeros(number_of_measurements_from_current_frame)
                        for index_of_idx, idx in enumerate(indices_of_measurements_exclusive):
                            association_vector[idx]=indices_of_tracks_exclusive[index_of_idx]
                        optimal_associations_all.append(association_vector)
                        cost_for_optimal_associations_non_exclusive = [0]
    
                    else:
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

                    for ith_optimal_option in range(len(optimal_associations_all)):
                        weight_of_global_hypothesis_in_log_format.append(-cost_for_optimal_associations_non_exclusive[ith_optimal_option]+np.sum(weight_for_missed_detection_hypotheses)+weight_of_exclusive_assosications+np.log(filter_predicted['globHypWeight'][global_hypothesis_index_from_pevious_frame])) # The global weight associated with this hypothesis
    
                    '''
                    Step 2.4.3 Generate the new global hypothesis based on the cost matrix of this previous global hypothesis
                    '''
                    globHyp_from_current_frame_under_this_globHyp_from_previous_frame=np.zeros((len(optimal_associations_all), number_of_detected_targets_from_previous_frame+number_of_measurements_from_current_frame))
                    '''
                    please refer to figure 2 of [1]. The following part discribes track 1 and 2 which is an
                    established Bernoulli track. 
                    The hypothesis indexing is the following:
                    h_* associated with measurement with indexing *, for instance h_2 means associated with measurement1, notice it is very important to get the indexing of measurement right.
                    '''
                    for track_index in range(number_of_detected_targets_from_previous_frame): # first  update the extablished Bernoulli components
                        single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index_from_pevious_frame][track_index] # Readout the single target hypothesis index as specified by the global hypothesis of previous step
                        indices_of_new_hypotheses_generated_from_this_previous_hypothesis = [idx for idx, value in enumerate(filter_updated['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame']) if value == single_target_hypothesis_index_specified_by_previous_step_global_hypothesis]
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
                    globWeight=[np.exp(x-maximum_weight_of_global_hypothesis_in_log_format) for x in weight_of_global_hypothesis_in_log_format]
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
        #cost_metrix_log = filter_updated['cost_matrix_log'] # row: tracks, column: measurements
        #weight_for_missed_detection_hypotheses = filter_updated['weight_for_missed_detection_hypotheses']
        
        # Initiate datastructure
        state_estimate = {}
        mean = []
        rotation = []
        elevation = []
        classification = []
        size = []
        covariance = []
        detection_score = []
        eB_list=[]
        association_history = [[] for i in range(number_of_tracks_at_current_frame)]
        association_counter=[]
        id=[]
        weight=[]

        if number_of_global_hypotheses_at_current_frame>0: # If there are valid global hypotheses
            highest_weight_global_hypothesis_index = np.argmax(globHypWeight) # get he index of global hypothesis with largest weight
            highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index] # get the global hypothesis with largest weight
            for track_index in range(len(highest_weight_global_hypothesis)): # number of tracks.
                single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index]) # Get the single target hypothesis index.

                if single_target_hypothesis_specified_by_global_hypothesis > -1: # If the single target hypothesis is not does not exist
                    #a =filter_updated['tracks'][track_index]['eB'] 
                    eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                    for clas in ['car','bicycle','truck','pedestrian','trailer','motorcycle','bus']:
                        birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(clas, parameters)
                        if eB >=extraction_thr and filter_updated['tracks'][track_index]['classificationB'][single_target_hypothesis_specified_by_global_hypothesis]==clas: # if the existence probability is greater than the threshold
                            mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis]) # then assume that this track exist.
                            rotation.append(filter_updated['tracks'][track_index]['rotationB'][single_target_hypothesis_specified_by_global_hypothesis])
                            elevation.append(filter_updated['tracks'][track_index]['elevationB'][single_target_hypothesis_specified_by_global_hypothesis])
                            classification.append(clas)
                            eB_list.append(eB)
                            id.append(filter_updated['tracks'][track_index]['idB'][single_target_hypothesis_specified_by_global_hypothesis])
                            covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_specified_by_global_hypothesis])
                            size.append(filter_updated['tracks'][track_index]['sizeB'][single_target_hypothesis_specified_by_global_hypothesis])
                            detection_score.append(filter_updated['tracks'][track_index]['detection_scoreB'][single_target_hypothesis_specified_by_global_hypothesis])
                            associated_measurement=filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_specified_by_global_hypothesis]
                            association_history[track_index].append(associated_measurement)
                            weight.append(filter_updated['tracks'][track_index]['giou'][single_target_hypothesis_specified_by_global_hypothesis])
                            association_counter.append(filter_updated['tracks'][track_index]['association_counter'][single_target_hypothesis_specified_by_global_hypothesis])


        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['rotation']=rotation
        state_estimate['elevation']=elevation
        state_estimate['size']=size
        state_estimate['classification']=classification
        state_estimate['id']=id
        state_estimate['detection_score'] = detection_score
        state_estimate['measurement_association_history'] = association_history
        state_estimate['eB']=eB_list
        state_estimate['weight']=weight
        state_estimate['association_counter']=association_counter

        return state_estimate

    def extractStates_with_custom_thr(self, filter_updated, thr):
        state_extraction_option = self.model['state_extraction_option']
        # Get data
        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypotheses_at_current_frame = len(globHypWeight)
        number_of_tracks_at_current_frame=len(filter_updated['tracks'])
        #cost_metrix_log = filter_updated['cost_matrix_log'] # row: tracks, column: measurements
        #weight_for_missed_detection_hypotheses = filter_updated['weight_for_missed_detection_hypotheses']
        
        # Initiate datastructure
        state_estimate = {}
        mean = []
        rotation = []
        elevation = []
        classification = []
        size = []
        covariance = []
        detection_score = []
        eB_list=[]
        association_all = []
        association_history_all=[[] for x in range(number_of_tracks_at_current_frame)]
        id=[]
        weight=[]

        if number_of_global_hypotheses_at_current_frame>0: # If there are valid global hypotheses
            highest_weight_global_hypothesis_index = np.argmax(globHypWeight) # get he index of global hypothesis with largest weight
            highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index] # get the global hypothesis with largest weight
            for track_index in range(len(highest_weight_global_hypothesis)): # number of tracks.
                single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index]) # Get the single target hypothesis index.
                if single_target_hypothesis_specified_by_global_hypothesis > -1: # If the single target hypothesis is not does not exist
                    #a =filter_updated['tracks'][track_index]['eB'] 
                    eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                    if eB >thr: # if the existence probability is greater than the threshold
                        mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis]) # then assume that this track exist.
                        rotation.append(filter_updated['tracks'][track_index]['rotationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        elevation.append(filter_updated['tracks'][track_index]['elevationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        classification.append(filter_updated['tracks'][track_index]['classificationB'][single_target_hypothesis_specified_by_global_hypothesis])
                        eB_list.append(eB)
                        id.append(filter_updated['tracks'][track_index]['idB'][single_target_hypothesis_specified_by_global_hypothesis])
                        covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis_specified_by_global_hypothesis])
                        size.append(filter_updated['tracks'][track_index]['sizeB'][single_target_hypothesis_specified_by_global_hypothesis])
                        detection_score.append(filter_updated['tracks'][track_index]['detection_scoreB'][single_target_hypothesis_specified_by_global_hypothesis])
                        associated_measurement=filter_updated['tracks'][track_index]['measurement_association_history'][single_target_hypothesis_specified_by_global_hypothesis]
                        association_all.append(associated_measurement)
                        association_history_all[track_index]=filter_updated['tracks'][track_index]['measurement_association_history']
                        
                        #if associated_measurement == -1:
                        #    likelihood_score = np.exp(weight_for_missed_detection_hypotheses[track_index])
                        #else:
                        #    likelihood_score = np.exp(cost_metrix_log[track_index][associated_measurement])
                        #weight.append(likelihood_score)
                        weight.append(filter_updated['tracks'][track_index]['giou'][single_target_hypothesis_specified_by_global_hypothesis])


        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['rotation']=rotation
        state_estimate['elevation']=elevation
        state_estimate['size']=size
        state_estimate['classification']=classification
        state_estimate['id']=id
        state_estimate['detection_score'] = detection_score
        state_estimate['measurement_association_history'] = association_history_all
        state_estimate['eB']=eB_list
        state_estimate['weight']=weight

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
            del filter_pruned['rotationPois'][idx-offset]
            del filter_pruned['elevationPois'][idx-offset]
            del filter_pruned['classificationPois'][idx-offset]
            del filter_pruned['sizePois'][idx-offset]
            del filter_pruned['idPois'][idx-offset]
            del filter_pruned['meanPois'][idx-offset]
            del filter_pruned['covPois'][idx-offset]
            del filter_pruned['detection_scorePois'][idx-offset]
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

        if len(globHyp_after_pruning)>0:

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
            if len(globHyp_after_pruning) > 0:
                tracks_to_be_removed = [x for x in range(len(globHyp_after_pruning[0])) if np.sum(globHyp_after_pruning[:,x]) == -len(globHyp_after_pruning)]
            else:
                tracks_to_be_removed=[]
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
                        del filter_pruned['tracks'][track_index]['rotationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['elevationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['classificationB'][index-offset]
                        del filter_pruned['tracks'][track_index]['sizeB'][index-offset]
                        del filter_pruned['tracks'][track_index]['idB'][index-offset]
                        del filter_pruned['tracks'][track_index]['detection_scoreB'][index-offset]
                        del filter_pruned['tracks'][track_index]['weight_of_single_target_hypothesis_in_log_format'][index-offset]
                        del filter_pruned['tracks'][track_index]['giou'][index-offset]
                        del filter_pruned['tracks'][track_index]['single_target_hypothesis_index_from_previous_frame'][index-offset]
                        del filter_pruned['tracks'][track_index]['measurement_association_history'][index-offset]
                        del filter_pruned['tracks'][track_index]['measurement_association_from_this_frame'][index-offset]
                        del filter_pruned['tracks'][track_index]['association_counter'][index-offset]
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