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
    filter_model['P']=np.diag([5,5,1,1])
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


def generate_validation_matrix(targets, unique_measurements_position):
    '''
    Generate validation matrix according to figure 1 and equation (3.5) of [2]. where the row index corresponds to measurements, and the column index corresponds to targets.
    Input arguments:
        targets: list of targets at current timestamp.
        unique_measurements_position: list of all the measurements in the gating regions of all targets at current timestamp.
    Output: 
        validation_matrix: a matrix represents which measurements fall into gating area of which targets. More info could be found in equation (3.5) of [2].
    '''
    num_target = len(targets)
    validation_matrix = np.zeros((len(unique_measurements_position), num_target+1)) # Initiate the validation matrix as an empty matrix.
    validation_matrix[:, 0] = 1 # Initiate the t_0 all be 1. t_0 stands for clutter.
    counter = 1
    for t in targets:
        valid_measurements_for_this_target_list = t.read_measurements_within_gating()
        valid_measurements_for_this_target =[(x['translation'][0], x['translation'][1]) for x in valid_measurements_for_this_target_list] # Convert from np.array to a list of tuple.
        if len(valid_measurements_for_this_target) != 0:
            for m in valid_measurements_for_this_target:
                index = unique_measurements_position.index(m)
                validation_matrix[index][counter] = 1 # set it to be 1
        counter += 1
    return validation_matrix

def generate_feasible_assiciation_matrices_of_joint_events(validation_matrix):
    '''
    Generate association matrix for every feasible joint event. See page 4 in [2] for detailed description of this process.

    Probablistic Data Association used all the measurements inside the gating area to construct a "virtual measurements", 
    whose moments are constrained to match that of all the other measurements. However, this style of information extraction 
    would result in double counting. For instance, if measurement_3 both belongs to target_1 and target_2, then this 
    measurement would be utilized twice.

    To solve this problem, joint probablistic data association is deviced and this step sits at the core of JPDA.
    The essense is a permutation problem. For each target t, there would be N_t measurments falls in the gating area.
    The joint events refers to the process that every target choose a single measurement in the gating area to form a
    possible association pair and combine all the association pair which could be happened at same time as a possible 
    feasible joint event.

    According to the description on page 4 in [2], this permutation process would carry out in the following fashion:
    1. scan the rows of validation_matrix and only pick one element per row.
    2. subject to the contraint that there can only be at most one element for each column, reflecting the fact that 
        there can only be one measurement associated with each target(This is the point target model assumption).
    3. the aforementioned constraint should not be imposed on t_0, which represent clutter.

    Input arguments:
        validation_matrix: a matrix(numpy array of array) which describes which measurements fall into gating regions of which targets.
    Output: 
        association_matrices: a list of association matrices correspond to all the feasible joint events.
    '''
    num_measurements = len(validation_matrix)           # Number of measurements in the gating areas of all targets.
    num_targets = len(validation_matrix[0])             # Number of targets + 1 which is t_0 represents clutter.

    association_matrices = [] # Iniitiate the list of association matrices, which will store the association matrices correspond to all the feasible joint events.
    for measurement_index in range(num_measurements):                   # For the index of every measurement, row_index.
        matrices_for_this_measurement = []
        measurement_vector = validation_matrix[measurement_index]       # Obtain the row vector of current row in the validation matrix.
        non_zero_target_index = [idx for idx, element in enumerate(measurement_vector) if element != 0 and idx != 0]    # Get all the indece of non zero targets for the current measurement m_(row_index).
        if measurement_index == 0:              # If association_matrices is still an empty list(it means we are still running into the first row now), we do the initialization as following:
            partial_union_matrix_one = np.zeros((num_measurements, num_targets))
            partial_union_matrix_one[measurement_index][0] = 1  # Stipulate that t_0 has 1 for current measurement m_1, this would exclude all other targets to be associated with the first measurement, m_1.
            # Add the matrix to matrices of current row.
            matrices_for_this_measurement.append(partial_union_matrix_one)
            for column_index in non_zero_target_index:  # For every index of non zero target for the current measurement m_1.
                partial_union_matrix_zero = np.zeros((num_measurements, num_targets))       # Initiate partial matrix with all 0, thus t_0 has 0.
                # For current measurement with "t_0 has 0" thus it is allowed to associate current measurement with one of the target, we could stipulate every such associatioin(according to validation matrix)
                # as one, and add the matrix to matrices of current row.
                partial_union_matrix_zero[measurement_index][column_index] = 1
                matrices_for_this_measurement.append(partial_union_matrix_zero)
            # Give all the association matrices already decided according to the information from validation matrix until current row, to association_matrices.
            association_matrices = copy.deepcopy(matrices_for_this_measurement)
        else:   # If association_matrices has been initiated, i.e. we are now running into second row or even deeper(third row, forth row, etc.).
            for previous_row_matrix_index in range(len(association_matrices)):   # Get index of every association matrix already decided according to the information from validation matrix until previous row.
                partial_union_matrix_one = association_matrices[previous_row_matrix_index].copy()   # Obtain each association matrix already decided according to the information from validation matrix until previous row.
                # The following operations will continue to update the existing association matrice decided according to the information from validation matrix until previous row, 
                # by checking the values of current row vector of validation matrix, to generate more permutations and corresponding association matrice until current row.
                partial_union_matrix_one[measurement_index][0] = 1 # Stipulate that t_0 has 1 for current measurement m_(row_index), this would exclude all other targets to be associated with the current measurement m_(row_index).
                matrices_for_this_measurement.append(partial_union_matrix_one)
                for column_index in non_zero_target_index:  # For every index of non zero target for the current measurement m_(row_index).
                    partial_union_matrix_zero = association_matrices[previous_row_matrix_index].copy()  # Obtain each association matrix already decided according to the information from validation matrix until previous row.
                    if sum(partial_union_matrix_zero[:, column_index]) == 0:    # If there has not been an association associated to this target, for the current association matrix already decided according to the information from validation matrix until previous row.
                        # We stipulate every such associatioin as one for the current measurement with "t_0 has 0", and add the matrix to matrices of current row.
                        partial_union_matrix_zero[measurement_index][column_index] = 1
                        matrices_for_this_measurement.append(partial_union_matrix_zero)
            # Give all the association matrices already decided according to the information from validation matrix until current row, to association_matrices.
            association_matrices = copy.deepcopy(matrices_for_this_measurement)
    return association_matrices

def construct_probability_dictionary(targets, unique_measurements_position):
    '''
    Construct a dictionary whose the keys are targets and the values are the Gaussian pdf likelihood probabilities of all measurements
    associated with every target(See equation (38) or (48) in [1].).
    
    Beware such likelihood probabilty information for every measurement in the gating region of every target has already been calculated 
    by function "compute_likelihood_probability_for_each_measurement" under class CVTarget, but only available for each target instance. 
    In order to ultilize such information to compute joint probability of every joint event denoted in the association_matrices(list of 
    association matrices correspond to all the feasible joint events), we have to reconstruct such info into a dictionary whose the keys 
    are targets and the values are the Gaussian pdf likelihood probabilities of all measurements associated with every target.
    
    Input arguments:
        targets: a list of tagets at current timestamp.
        unique_measurements_position: list of all the measurements in the gating regions of all targets at current timestamp.
    Output: 
        probability_dictionary: a dict whose key is target and the corresponding value is the likelhood probabilities of all measurements in
            the gating regoin of this target.
    '''
    probability_dictionary = {} # Initiate an empty dictionary for joint probabilities, where the key is target_position and value probability vector.
    for t in targets:
        target_position = (t.target_state[0][0],t.target_state[1][0])    # Generate the key, which is target position.
        measurements_in_gating_area_list = t.read_measurements_within_gating() # Read out the measurements within gating area of current target.
        measurements_in_gating_area = [(x['translation'][0], x['translation'][1]) for x in measurements_in_gating_area_list] # Convert it into an array of sets.
        probability_measurement_associated_with_this_target = t.read_probability_for_each_measurement() # Read the likelihood probability of each measurement within the gating area of current target.
        probability_vector = np.zeros(len(unique_measurements_position)) # Initiate the probability vector, which has the length of len(unique_measurements)
        for measurement in measurements_in_gating_area:     # For every measurement in the gating region of current target:
            idx = unique_measurements_position.index(measurement) # Get the index of this measurement along the probability vector
            # Get the Gaussian pdf likelihood of this measurement as seen from this target.
            probability_vector[idx] = probability_measurement_associated_with_this_target[measurement]
        probability_dictionary[target_position] = probability_vector # Pair the Gaussian pdf likelihoods of all measurements in the gating region of current target with current target.
    return probability_dictionary

def compute_joint_probabilities(targets, feasible_association_matrices_of_joint_events, probability_dictionary, P_D, clutter_intensity):
    '''
    Compute joint probability based on association matrix of feasible joint event and probability dictionary for each joint event, 
    according to equation 3.18 of [2].
    Here are some key notations and information in order to undertand Eq. 3.18: 
    Tao_j is association indicator for measurement j as defined by Eq. 3.3 in [2]
    Delta_t is target detection indicator for target t as defined by Eq. 3.4 in [2]
    c normalization constant is the summation of probabilities of all joint events.
    C_phi is the exponatial phi of C, where C is the clutter density/intensity(C = number_of_clutter_per_frame/FOV_area). However, because the 
        magnitude of C (in units of 1/volume in the measurement space/FOV) is quite variable and phi can be 10 or more. The problem can be 
        avoided simply by letting 1/C be the unit volume in calculating (3.18), so that C_phi is replaced by l_phi(thus equal to 1). This change 
        cancels out in the exponential factor, and it causes the denominator of "Gaussian pdf likelihood probability of selected measurement-target 
        pair in this joint event at current frame" to be multiplied by C. In another word, when calculate equation (3.18), every "Gaussian pdf likelihood 
        probability of selected measurement-target pair in this joint event at current frame" needs to be divided by C. We can also see the same
        details in equation (47) of [1].

    each matrix in feasible_association_matrices_of_joint_events represent a possible joint event. measurement_vector[0]
    reperesent t_0. If it is indicated as 0 means this measurement originates from a target. otherwise, it means this measurement
    orginates from a clutter. 

    Input arguments:
        targets: a list of tagets at current timestamp.
        feasible_association_matrices_of_joint_events: a list of association matrices correspond to all the feasible joint events.
        probability_dictionary: a dict whose key is target and the corresponding value is the likelhood probabilities of all measurements in
            the gating regoin of this target.
        P_D: detection probability.
    Output: 
        joint_probabilities: a dictionary with the feasible joint event index as the key and joint_probability as value.
    '''
    joint_probabilities = {} 
    num_matrices = len(feasible_association_matrices_of_joint_events)           # The number of feasible joint events at current frame.
    sum_joint_probabilities = 0     # The summation of probabilities of all joint events. Beware this is the "c, normalization constant" in equation (3.18) of [2].

    for joint_event_index in range(num_matrices): # Loop over all the feasible joint event matrices.
        matrix_of_joint_events = feasible_association_matrices_of_joint_events[joint_event_index]    # Get the association matrix representation of every feasible joint event.
        joint_probability = 1
        target_idx = 0
        for t in targets:
            target_idx += 1 # starting from t_1.
            target_position = (t.target_state[0][0], t.target_state[1][0])
            if (matrix_of_joint_events[:, target_idx] == 0).all(): # If there is no measurement associated with current target, then current target is not detected.
                joint_probability *= (1 - P_D) # Multiply with (1 - P_D) as current target is miss detected. See the last part of equation (3.18) of [2]. 
            else:
                target_vector = matrix_of_joint_events[:, target_idx] # Get the column vector associated with current target.
                measurement_index = np.argmax(target_vector) # Get the index of measurement associated with current target(the corresponding element should be 1).
                # Obtain the Gaussian pdf likelihood probabilities of all the measurements in the gating regoin for associated target.
                likelihood_probabilities_of_all_measurements_in_gating_area_of_current_target = probability_dictionary[target_position]
                # Select the Gaussian pdf likelihood probability of selected measurement-target pair in this joint event at current frame,
                # multiply with the detection probability. See the first part of equation (3.18) of [2] or the first three parts of equation (47) of [1].
                joint_probability *= (P_D * likelihood_probabilities_of_all_measurements_in_gating_area_of_current_target[measurement_index] / clutter_intensity)
        # Obtain joint probability for each feasible joint event.
        joint_probabilities[joint_event_index] = joint_probability
        sum_joint_probabilities = sum_joint_probabilities + joint_probabilities[joint_event_index]
    
    # Normalize the every probability of joint event by dividing of "the summation of probabilities of all joint events(c, normalization constant in equation (3.18) of [2])".
    for joint_event_index in range(num_matrices):
        joint_probabilities[joint_event_index] = joint_probabilities[joint_event_index]/sum_joint_probabilities
    return joint_probabilities

def compute_marginal_probability(targets, feasible_association_matrices_of_joint_events, joint_probabilities, unique_measurements_position): # for detailed description please refer to Eq. 3.19 of [2]
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
    marginal_probability = {} # Initiate Marginal Probability to be an empty dictionary.
    num_matrices = len(feasible_association_matrices_of_joint_events)       # The number of feasible joint events at current frame.
    t_index = 0
    for t in targets:
        t_index += 1 # starting from position one since t_0 is clutter
        t_position = (t.target_state[0][0],t.target_state[1][0])
        for m in t.measurements_within_gating:  # Loop over all measurements in the gating region of current target:
            beta = 0  # Initiate Beta to be zero.
            m_position = (m['translation'][0], m['translation'][1])
            measurement_idx = unique_measurements_position.index(m_position) 
            for joint_event_index in range(num_matrices):   # Loop over all feasible association events, for current measurement and current target:
                # Obtain the association matrix for current joint event.
                matrix_of_joint_events = feasible_association_matrices_of_joint_events[joint_event_index]
                if matrix_of_joint_events[measurement_idx][t_index] == 1: # If the association matrix for current joint event indicate that this measurement is associated with this target:
                    # The probability of this association is added to the marginal probability of this measurement. (Since we need to sum over all joint probabilities whose corresponding 
                    # element represents "association between current measurement and current target" in the current association matrix is 1. See equation (3.19) in [2].)
                    beta += joint_probabilities[joint_event_index]
            marginal_probability[(t_position, m_position)] = beta # Associate key to the value.
    return marginal_probability

def compute_marginals(targets): # for detailed description please refer to Eq. 3.19 of [2]
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
    marginal_probability = {} # Initiate Marginal Probability to be an empty dictionary.    
    t_index = 0
    for t in targets:
        t_index += 1 # starting from position one since t_0 is clutter
        t_position = (t.target_state[0][0],t.target_state[1][0])
        for m in t.measurements_within_gating:  # Loop over all measurements in the gating region of current target:
            m_position = (m['translation'][0], m['translation'][1])
            likelihood=t.likelihood_probability_for_each_measurement[m_position]
            marginal_probability[(t_position, m_position)] = likelihood # Associate key to the value.
    return marginal_probability
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

    # Kalman update with JPDA.
    def jpda_update(self, marginal_probability, Z):
        zpred = self.zpred() # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = [] # innovation is defined by equation (2.5) in [2]
        betas = []
        t_position = (self.target_state[0][0], self.target_state[1][0])     # Position of current target.
        for z in Z: 
            
            innovations.append(np.array([z['translation'][0] - zpred[0],z['translation'][1] - zpred[1]]).reshape(-1,1).astype(np.float64)) # See eqation (2.14) in [2]
            m_position = (z['translation'][0], z['translation'][1])     # Position of current measurement which is in the gating region of current target.
            # Obtain the marginalized association probability between target t and measurement j (in the gating area of target t), beta_j^t with j = 0, 1, 2, ...
            betas.append(marginal_probability[t_position, m_position])
         
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association".
        # According to equation. 3.20 in [2].
        beta_0 = 1 - np.sum(betas)

        # Reduce the mixture into a combined innovation weighted by betas(association probabilities).
        combined_innovation = np.zeros_like(zpred)
        for j, innovation_for_each_measurement in enumerate(innovations):
            combined_innovation += betas[j] * innovation_for_each_measurement           # Calculate the combined innovation, v(k). See equation (40) in [1].
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)   # Calculate the Kalman gain, W(k). See equation (41) in [1].
        self.target_state = self.target_state + W@combined_innovation            # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].
        #print(self.target_state)
        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j]*innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation@combined_innovation.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov
        #print(self.P)
    # Kalman update with JPDA.
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

    def pda_update(self, Z):
        zpred = self.zpred()        # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = []
        betas_unorm = []
        for z_idx, z in enumerate(Z):
                # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
            # Calculate innovation for each measuremnt, (z - zpred), which is v_i(k) in equation (40) in [1].
            innovations.append(np.array([z['translation'][0] - zpred[0],z['translation'][1] - zpred[1]]).reshape(-1,1).astype(np.float64)) # See eqation (2.14) in [2]
            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) in [1].
            betas_unorm.append(self.P_D * gauss(z['translation'], zpred, self.S)/self.lambda_clutter)
        # Calculate the probablity of "association between every measurement(in the gating area, if gating is performed before update) and current target is the correct association", beta_i(k).
        # See equation (37) in [1].
        betas = betas_unorm / (np.sum(betas_unorm) + (1 - self.P_D))
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association". See equation (37) in [1]. 
        beta_0 = (1 - self.P_D) / (np.sum(betas_unorm) + (1 - self.P_D))

        # Reduce the mixture into a combined innovation weighted by betas(association probabilities).
        combined_innovation = np.zeros_like(zpred)
        for j, innovation_for_each_measurement in enumerate(innovations):
            # Calculate the combined innovation, v(k). See equation (40) in [1].
            combined_innovation += betas[j] * innovation_for_each_measurement

        # Calculate the Kalman gain, W(k). See equation (41) in [1].
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)

        # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].
        self.target_state = self.target_state + W@combined_innovation
        #print(self.target_state)

        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j]*innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation @
                        combined_innovation.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov
        #print(self.P)

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
            #mah=np.transpose(position_difference)@np.linalg.inv(self.S)@(position_difference)
            if mah < self.eta:
            #if mah < gating_thr:
            #euc=(z['translation'][0] - zpred[0])**2+(z['translation'][1] - zpred[1])**2
            #if euc < self.eta:
                #print(position_difference)
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

    def compute_likelihood_probability_for_each_measurement(self, measurements):
        '''
        Generate a dictionary whose key is the measurement and the corresponding value is its likelihood, See equation (38) or (48) in [1].
        Seeing from this particular target return a dictionary [measurement]: likelihood.
        '''
        zpred=self.zpred()
        for m in measurements:

            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) or (48) in [1].
            self.likelihood_probability_for_each_measurement[(m['translation'][0], m['translation'][1])] = self.P_D * gauss(m['translation'], zpred, self.S)

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

    def new(self, new_target):
        x=new_target['translation'][0]
        y=new_target['translation'][1]
        elevation=new_target['translation'][2]
        rotation=new_target['rotation']
        size=new_target['size']
        velocity=new_target['velocity']
        classification=new_target['detection_name']
        detection_score=new_target['detection_score']
        return Target(x, y, elevation, rotation, size, velocity, classification, detection_score, self.F, self.H, self.P, self.Q, self.R, self.eta, self.P_D, self.lambda_clutter, self.birth_initiation,self.death_initiation)