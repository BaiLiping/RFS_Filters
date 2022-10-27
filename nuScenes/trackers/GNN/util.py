from cv2 import Mahalanobis
import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy.stats import poisson
import math
import numpy.matlib
import matplotlib.pyplot as plt
import time
from functools import reduce
import operator
from scipy.spatial import distance
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def gen_filter_model(classification, p_D, average_number_of_clutter_per_frame, death_counter_kill,birth_counter_born, birth_initiation, death_initiation,gating_threshold):
    P = {
        'bicycle': {'x': 0.05390982, 'y': 0.05039431, 'z': 0.01863044, 'yaw': 1.29464435,
                    'l': 0.02713823, 'w': 0.01169572, 'h': 0.01295084,
                    'dx': 0.04560422, 'dy': 0.04097244, 'dz': 0.01725477, 'dyaw': 1.21635902},
        'bus': {'x': 0.17546469, 'y': 0.13818929, 'z': 0.05947248, 'yaw': 0.1979503,
                'l': 0.78867322, 'w': 0.05507407, 'h': 0.06684149,
                'dx': 0.13263319, 'dy': 0.11508148, 'dz': 0.05033665, 'dyaw': 0.22529652},
        'car': {'x': 0.08900372, 'y': 0.09412005, 'z': 0.03265469, 'yaw': 1.00535696,
                'l': 0.10912802, 'w': 0.02359175, 'h': 0.02455134,
                'dx': 0.08120681, 'dy': 0.08224643, 'dz': 0.02266425, 'dyaw': 0.99492726},
        'motorcycle': {'x': 0.04052819, 'y': 0.0398904, 'z': 0.01511711, 'yaw': 1.06442726,
                       'l': 0.03291016, 'w': 0.00957574, 'h': 0.0111605,
                       'dx': 0.0437039, 'dy': 0.04327734, 'dz': 0.01465631, 'dyaw': 1.30414345},
        'pedestrian': {'x': 0.03855275, 'y': 0.0377111, 'z': 0.02482115, 'yaw': 2.0751833,
                       'l': 0.02286483, 'w': 0.0136347, 'h': 0.0203149,
                       'dx': 0.04237008, 'dy': 0.04092393, 'dz': 0.01482923, 'dyaw': 2.0059979},
        'trailer': {'x': 0.23228021, 'y': 0.22229261, 'z': 0.07006275, 'yaw': 1.05163481,
                    'l': 1.37451601, 'w': 0.06354783, 'h': 0.10500918,
                    'dx': 0.2138643, 'dy': 0.19625241, 'dz': 0.05231335, 'dyaw': 0.97082174},
        'truck': {'x': 0.14862173, 'y': 0.1444596, 'z': 0.05417157, 'yaw': 0.73122169,
                  'l': 0.69387238, 'w': 0.05484365, 'h': 0.07748085,
                  'dx': 0.10683797, 'dy': 0.10248689, 'dz': 0.0378078, 'dyaw': 0.76188901}
    }

    Q = {
        'bicycle': {'x': 1.98881347e-02, 'y': 1.36552276e-02, 'z': 5.10175742e-03, 'yaw': 1.33430252e-01,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 1.98881347e-02, 'dy': 1.36552276e-02, 'dz': 5.10175742e-03, 'dyaw': 1.33430252e-01},
        'bus': {'x': 1.17729925e-01, 'y': 8.84659079e-02, 'z': 1.17616440e-02, 'yaw': 2.09050032e-01,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.17729925e-01, 'dy': 8.84659079e-02, 'dz': 1.17616440e-02, 'dyaw': 2.09050032e-01},
        'car': {'x': 1.58918523e-01, 'y': 1.24935318e-01, 'z': 5.35573165e-03, 'yaw': 9.22800791e-02,
                'l': 0, 'w': 0, 'h': 0,
                'dx': 1.58918523e-01, 'dy': 1.24935318e-01, 'dz': 5.35573165e-03, 'dyaw': 9.22800791e-02},
        'motorcycle': {'x': 3.23647590e-02, 'y': 3.86650974e-02, 'z': 5.47421635e-03, 'yaw': 2.34967407e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.23647590e-02, 'dy': 3.86650974e-02, 'dz': 5.47421635e-03, 'dyaw': 2.34967407e-01},
        'pedestrian': {'x': 3.34814566e-02, 'y': 2.47354921e-02, 'z': 5.94592529e-03, 'yaw': 4.24962535e-01,
                       'l': 0, 'w': 0, 'h': 0,
                       'dx': 3.34814566e-02, 'dy': 2.47354921e-02, 'dz': 5.94592529e-03, 'dyaw': 4.24962535e-01},
        'trailer': {'x': 4.19985099e-02, 'y': 3.68661552e-02, 'z': 1.19415050e-02, 'yaw': 5.63166240e-02,
                    'l': 0, 'w': 0, 'h': 0,
                    'dx': 4.19985099e-02, 'dy': 3.68661552e-02, 'dz': 1.19415050e-02, 'dyaw': 5.63166240e-02},
        'truck': {'x': 9.45275998e-02, 'y': 9.45620374e-02, 'z': 8.38061721e-03, 'yaw': 1.41680460e-01,
                  'l': 0, 'w': 0, 'h': 0,
                  'dx': 9.45275998e-02, 'dy': 9.45620374e-02, 'dz': 8.38061721e-03, 'dyaw': 1.41680460e-01}
    }

    R = {
        'bicycle': {'x': 0.05390982, 'y': 0.05039431, 'z': 0.01863044, 'yaw': 1.29464435,
                    'l': 0.02713823, 'w': 0.01169572, 'h': 0.01295084},
        'bus': {'x': 0.17546469, 'y': 0.13818929, 'z': 0.05947248, 'yaw': 0.1979503,
                'l': 0.78867322, 'w': 0.05507407, 'h': 0.06684149},
        'car': {'x': 0.08900372, 'y': 0.09412005, 'z': 0.03265469, 'yaw': 1.00535696,
                'l': 0.10912802, 'w': 0.02359175, 'h': 0.02455134},
        'motorcycle': {'x': 0.04052819, 'y':0.0398904, 'z': 0.01511711, 'yaw': 1.06442726,
                       'l': 0.03291016, 'w':0.00957574, 'h': 0.0111605},
        'pedestrian': {'x': 0.03855275, 'y': 0.0377111, 'z': 0.02482115, 'yaw': 2.0751833,
                       'l': 0.02286483, 'w': 0.0136347, 'h': 0.0203149},
        'trailer': {'x': 0.23228021, 'y': 0.22229261, 'z': 0.07006275, 'yaw': 1.05163481,
                    'l': 1.37451601, 'w': 0.06354783, 'h': 0.10500918},
        'truck': {'x': 0.14862173, 'y': 0.1444596, 'z': 0.05417157, 'yaw': 0.73122169,
                  'l': 0.69387238, 'w': 0.05484365, 'h': 0.07748085}
    }

    filter_model = {}  # filter_model is the dictionary which has all the corresponding parameters of the generated filter_model

    T = 1  # Sampling period, time step duration between two scans(frames).
    filter_model['T'] = T
    # Dynamic motion filter_model parameters(The motion used here is Constant Velocity (CV) filter_model):
    # State transition matrix, F_k.
    # F_k = np.array([
    #         [1, 0, T_s,   0],
    #         [0, 1,   0, T_s],
    #         [0, 0,   1,   0],
    #         [0, 0,   0,   1],
    #     ])
    filter_model['F'] = np.eye(4, dtype=np.float64)
    I = T*np.eye(2, dtype=np.float64)
    filter_model['F'][0:2, 2:4] = I
    #sigma_v = 1     # Standard deviation of the process noise.
    #Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    #Q = np.zeros((4, 4), dtype=np.float64)
    #Q[np.ix_([0, 2], [0, 2])] = Q1
    #Q[np.ix_([1, 3], [1, 3])] = Q1
    #filter_model['Q'] = sigma_v ** 2 * Q  # Covariance of process noise
    filter_model['Q']=np.diag([Q[classification]['x'], Q[classification]['y'],Q[classification]['dx'], Q[classification]['dy']])
    #filter_model['P']=np.diag([P[classification]['x'], P[classification]['y'],P[classification]['dx'], P[classification]['dy']])
    filter_model['P']=np.diag([10,10,1,1])
    filter_model['death_counter_kill']=death_counter_kill
    filter_model['birth_counter_born']=birth_counter_born
    filter_model['birth_initiation']=birth_initiation
    filter_model['death_initiation']=death_initiation

    # Observation/Measurement filter_model parameters (noisy x and y only rather than v_x, v_y):
    filter_model['H'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation filter_model matrix.
    #filter_model['R'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    filter_model['R']=np.diag([R[classification]['x'], R[classification]['y']])

    # Measurements parameters. See equation (20) in [1].
    filter_model['P_D'] = p_D  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D)

    # Compute clutter intensity lambda_c. See equation (47) and corresponding explanation in [1]. 
    filter_model['average_number_of_clutter_per_frame']=average_number_of_clutter_per_frame
    x_range = [-60, 60]  # X range of measurements
    y_range = [-60, 60]  # Y range of measurements
    # Compute and store Area
    filter_model['area'] = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area.
    clutterIntensity = average_number_of_clutter_per_frame/filter_model['area']  # Generate clutter intensity (clutter intensity lambda_c = lambda_t/A)
    
    filter_model['clutterIntensity'] = clutterIntensity
    filter_model['unitclutterIntensity'] = 1/filter_model['area']

    # gating threshold
    filter_model['eta'] = gating_threshold
    return filter_model


# The probability density function (pdf) of the d-dimensional multivariate normal distribution
def mvnpdf(x, mean, covariance):
    # x = np.array(x, dtype=np.float64)
    # mean = np.array(mean, dtype=np.float64)
    # covariance = np.array(covariance, dtype=np.float64)
    d = mean.shape[0]
    delta_m = x - mean
    pdf_res = 1.0/(np.sqrt((2*np.pi)**d *np.linalg.det(covariance))) * np.exp(-0.5*np.transpose(delta_m).dot(np.linalg.inv(covariance)).dot(delta_m))[0][0]
    # pdf_res = 1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) * math.exp(-0.5 * np.transpose(delta_m).dot(np.linalg.inv(covariance)).dot(delta_m))

    return pdf_res


def gauss(x: np.array, mu: np.array, P: np.array) -> np.float:
    '''
    Generate a Guassian probability distribution function.
    '''
    #n = mu.shape[1]
    n=2
    Pinv = np.linalg.inv(P)
    #x_new=np.array([np.array([x[0]]),np.array([x[1]])])
    
    x_new=np.array([x[0], x[1]]).reshape(-1,1).astype(np.float64)
    #difference=np.array([np.array(x[0]-mu[0][0]),np.array(x[1]-mu[1][0])])
    #res = ((1/(np.sqrt((2*np.pi))**n)) * np.exp((-1 / 2) * difference.T@Pinv@difference))

    res = ((1/(np.sqrt((2*np.pi))**n)) * np.exp((-1 / 2) * (x_new - mu).T@Pinv@(x_new - mu)))[0][0]
    return res


def compute_cost_matrix(targets, Z_k): # for detailed description please refer to Eq. 3.19 of [2]
    '''
    Compute marginal probability based on feasible association matrix of this measurement and the joint probability as indicated by Eq. 3.19 in [2].
    The marginal association probability of measurement j is associated with target t is the summation all the joint probabilities(joint events) which inlcude this association.

    Input arguments:
        targets: a list of targets
        feasible_association_matrices_of_joint_events: a list of matrices each indicating a joint event
        joint_probabilities: a dictionary for joint probabilities
        unique_measurements_position: all the measurements associated with targets
    Output:
        marginal_probability with (t_position, m_position) pair as key and the marginal probability between this measurement and this target as value.
    ''' 
    n_targets = len(targets)
    n_m = len(Z_k)
    cost_matrix = -100*np.ones((n_m, n_targets))
    for t_idx, t in enumerate(targets):
        t_position = (t.target_state[0][0],t.target_state[1][0])
        for m in t.measurements_within_gating:
            m_position = (m['translation'][0], m['translation'][1])
            likelihood=t.likelihood_probability_for_each_measurement[m_position]
            for z_idx, z in enumerate(Z_k):  # Loop over all measurements in the gating region of current target:
                z_position = (z['translation'][0], z['translation'][1])
                if z_position == m_position:
                    cost_matrix[z_idx][t_idx]=likelihood
    return cost_matrix


"For one constant velocity target, perform PDA once per frame."
class Target:
    def __init__(self,
                 id,
                 x,
                 y,
                 elevation,
                 rotation,
                 size,
                 velocity,
                 classification,
                 detection_score,
                 F: np.array,
                 H: np.array,
                 P,
                 Q: np.array,
                 R: np.array,
                 eta: np.float,
                 P_D: np.float,
                 lambda_clutter: np.float,
                 birth_initiation: np.int,
                 death_initiation):
        # lambda_clutter is spatial density of clutter under Poisson clutter model(Thus in this code we use parametric PDA, see equation (37) in [1].).
        self.lambda_clutter = lambda_clutter    # self.lambda_clutter is spatial density of false measurements(clutter).
        self.id=id
        self.target_state = np.array([
            [x],
            [y],
            [velocity[0]],
            [velocity[1]],
        ])                                     # self.target_state represents the state of each target.
        self.elevation=elevation
        self.rotation=rotation
        self.size=size
        self.velocity=velocity
        self.classification=classification
        self.detection_score=detection_score
        self.F = F                                          # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
        self.H = H                                          # H is measurement matrix which represents the measurement model in update step.
        self.Q = Q                                          # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R                                          # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.P = P                             # State covariance matrix, P, is initialized using 4-by-4 diagonal matrix for the current target
        self.S = self.innov_cov(self.H, self.P, self.R)     # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        self.eta = eta
        
        
        self.P_D = P_D  # P_D is the target detection probability.
        self.birth_counter = birth_initiation   # when the target is initiated, counter is set at one                                 
        self.death_counter = death_initiation   # initiate a death counter
        self.measurements_within_gating = []

        self.likelihood_probability_for_each_measurement = {} # a dictionary where the key is the position of each measurement and value its likelihood as seen from this target
    
    def read_target(self):
        target={}
        target['id']=self.id
        target['translation']=self.target_state
        target['elevation']=self.elevation
        target['size']=self.size
        target['velocity']=self.velocity
        target['rotation']=self.rotation
        target['classification']=self.classification
        target['detection_score']=self.detection_score
        return target
    @staticmethod
    def innov_cov(H, P, R):
        # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        return H@P@np.transpose(H) + R

    # Kalman prediction. 
    def predict(self):
        # Calculate all the variables regarding motion model.
        self.target_state = self.F@self.target_state            # Equation (30) in [2].
        self.P = self.F@self.P@np.transpose(self.F) + self.Q    # Equation (32) in [2].
        #print(self.P)
        self.S = self.innov_cov(self.H, self.P, self.R)         # Equation (33) in [2].

    def kalman_update(self, measurement):
        zpred = self.zpred() # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        t_position = (self.target_state[0][0], self.target_state[1][0])     # Position of current target.
        innovations=np.array([measurement['translation'][0] - zpred[0],measurement['translation'][1] - zpred[1]]).reshape(-1,1).astype(np.float64) # See eqation (2.14) in [2]
         
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association".
        # According to equation. 3.20 in [2].

        K = self.P@np.transpose(self.H)@np.linalg.inv(self.S)   # Calculate the Kalman gain, W(k). See equation (41) in [1].
        self.target_state = self.target_state + K@innovations            # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].
        #print(self.target_state)
        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].

        #sprd_innov = W@(beta_boi - innovations@innovations.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - K@self.S@K.T
        self.elevation=measurement['translation'][2]
        self.rotation=measurement['rotation']
        self.size=measurement['size']
        self.velocity=measurement['velocity']
        self.classification=measurement['detection_name']
        self.detection_score=measurement['detection_score']

        #print(self.P)

    def compute_likelihood_probability_for_each_measurement(self, measurements):
        '''
        Generate a dictionary whose key is the measurement and the corresponding value is its likelihood, See equation (38) or (48) in [1].
        Seeing from this particular target return a dictionary [measurement]: likelihood.
        '''
        zpred=self.zpred()
        for m in measurements:

            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) or (48) in [1].
            self.likelihood_probability_for_each_measurement[(m['translation'][0], m['translation'][1])] = self.P_D * gauss(m['translation'], zpred, self.S)

    def zpred(self):
        # Predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        zpred = self.H@self.target_state
        #zpred = np.array([[zpred[0][0]], [zpred[1][0]]])
        return zpred
    
    def gating(self,gating_thr, Z_k):
        zpred = self.zpred()  # Calculate predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        # Only keep the measurements within the gating area for currect target, according to equation (34) in [1].
        
        for z_index, z in enumerate(Z_k):
            #position_difference=np.array([z['translation'][0] - zpred[0],z['translation'][1] - zpred[1]]).reshape(-1,1).astype(np.float64)
            measurement_position=np.array([z['translation'][0],z['translation'][1]]).reshape(-1,1)
            predicted_position=zpred
            mah = distance.mahalanobis(measurement_position, predicted_position, np.linalg.inv(self.S))
            #translation = [self.target_state[0][0],self.target_state[1][0], self.elevation] 
            #self_bbox = Box(translation, self.size, Quaternion(self.rotation))
            #self_bottom_corners=self_bbox.bottom_corners()
            #m_bbox = Box(z['translation'], z['size'], Quaternion(z['rotation']))
            #m_bottom_corners=m_bbox.bottom_corners()
            #self_corner = [self_bottom_corners[0][0],self_bottom_corners[0][1], self_bottom_corners[3][0],self_bottom_corners[3][1]]
            #m_corner = [m_bottom_corners[0][0],m_bottom_corners[0][1], m_bottom_corners[3][0],m_bottom_corners[3][1]]
            #Giou = Giou(self_corner, m_corner)
            #Ciou = Ciou(self_corner, m_corner)
            #Diou = Diou(self_corner, m_corner)
            #if self.distance == Giou:

            #    if Giou < self.eta:
                #mah=np.transpose(position_difference)@np.linalg.inv(self.S)@(position_difference)
                #if mah < self.eta:
                #if mah < gating_thr:
                #euc=(z['translation'][0] - zpred[0])**2+(z['translation'][1] - zpred[1])**2
                #if euc < self.eta:
                    #print(position_difference)
            #        self.measurements_within_gating.append(z)
            #elif self.distance == Ciou:
            #    if Ciou < self.eta:
            #        self.measurements_within_gating.append(z)
            #elif self.distance == Mahalanobis:
            if mah < self.eta:
                self.measurements_within_gating.append(z)

        return self.measurements_within_gating


    '''
    def gating(self,gating_thr, Z_k):
        zpred = self.zpred()  # Calculate predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        # Only keep the measurements within the gating area for currect target, according to equation (34) in [1].
        
        for z_index, z in enumerate(Z_k):
            position_difference=np.array([z['translation'][0] - zpred[0],z['translation'][1] - zpred[1]]).reshape(-1,1).astype(np.float64)
            mah=np.transpose(position_difference)@np.linalg.inv(self.S)@(position_difference)
            if mah < self.eta:
            #if mah < gating_thr:
            #euc=(z['translation'][0] - zpred[0])**2+(z['translation'][1] - zpred[1])**2
            #if euc < self.eta:
                #print(position_difference)
                self.measurements_within_gating.append(z)

        return self.measurements_within_gating
    '''

    def increase_birth_counter(self): # when a potential track has measurement associated, update counter
        self.birth_counter += self.detection_score
    def decrease_birth_counter(self):
        self.birth_counter -= self.detection_score
    def read_birth_counter(self): # return the birth counter
        return self.birth_counter
    def increase_death_counter(self): # when a potential death track lacks data association, update death counter
        self.death_counter +=self.detection_score
    def decrease_death_counter(self):
        self.death_counter -=self.detection_score
    def read_death_counter(self): #return death counter
        return self.death_counter        
      
    def clear_measurements_association(self): # reset measurements association
        self.measurements_within_gating = []


    def read_measurements_within_gating(self): # return the associated measurements
        return self.measurements_within_gating
    
    def read_probability_for_each_measurement(self):
        return self.likelihood_probability_for_each_measurement


"Constant Velocity Target Maker, it is just kind of wrapper to call CVTarget, which runs the core part of Kalman + JPDA."
class TargetMaker:

    def __init__(self, filter_model):
        self.lambda_clutter = filter_model['clutterIntensity']
        self.Q = filter_model['Q']              # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = filter_model['R']
        T=filter_model['T']              # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])                      # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in the prediction step.
        self.eta = filter_model['eta']
        self.P_D = filter_model['P_D']          # P_D is the target detection probability.
        self.birth_initiation = filter_model['birth_initiation']
        self.death_initiation = filter_model['death_initiation']
        self.H=filter_model['H']
        self.P=filter_model['P']

    def new(self, new_target, target_id):
        x=new_target['translation'][0]
        y=new_target['translation'][1]
        elevation=new_target['translation'][2]
        rotation=new_target['rotation']
        size=new_target['size']
        #velocity=new_target['velocity']
        velocity=[0,0]
        classification=new_target['detection_name']
        detection_score=new_target['detection_score']
        id=target_id
        return Target(id, x, y, elevation, rotation, size, velocity, classification, detection_score, self.F, self.H, self.P, self.Q, self.R, self.eta, self.P_D, self.lambda_clutter, self.birth_initiation,self.death_initiation)