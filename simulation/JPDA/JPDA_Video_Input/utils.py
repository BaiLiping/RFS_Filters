import numpy as np
import copy


def gauss(x: np.array, mu: np.array, P: np.array) -> np.float:
    '''
    Generate a Guassian probability distribution function.
    '''
    n = mu.shape[1]
    Pinv = np.linalg.inv(P)
    res = ((1/(np.sqrt((2*np.pi))**n)) * np.exp((-1 / 2) * (x - mu).T@Pinv@(x - mu)))[0][0]
    return res


def generate_validation_matrix(targets, unique_measurements):
    '''
    Generate validation matrix according to figure 1 and equation (3.5) of [2]. where the row index corresponds to measurements, and the column index corresponds to targets.
    Input arguments:
        targets: list of targets at current timestamp.
        unique_measurements: list of all the measurements in the gating regions of all targets at current timestamp.
    Output: 
        validation_matrix: a matrix represents which measurements fall into gating area of which targets. More info could be found in equation (3.5) of [2].
    '''
    num_target = len(targets)
    validation_matrix = np.zeros((len(unique_measurements), num_target+1)) # Initiate the validation matrix as an empty matrix.
    validation_matrix[:, 0] = 1 # Initiate the t_0 all be 1. t_0 stands for clutter.
    counter = 1
    for t in targets:
        valid_measurements_for_this_target = t.read_measurements_within_gating()
        valid_measurements_for_this_target =[(x[0][0], x[1][0]) for x in valid_measurements_for_this_target] # Convert from np.array to a list of tuple.
        if len(valid_measurements_for_this_target) != 0:
            for m in valid_measurements_for_this_target:
                index = unique_measurements.index(m)
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


def construct_probability_dictionary(targets, unique_measurements):
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
        unique_measurements: list of all the measurements in the gating regions of all targets at current timestamp.
    Output: 
        probability_dictionary: a dict whose key is target and the corresponding value is the likelhood probabilities of all measurements in
            the gating regoin of this target.
    '''
    probability_dictionary = {} # Initiate an empty dictionary for joint probabilities, where the key is target_position and value probability vector.
    for t in targets:
        target_position = (t.target_state[0][0],t.target_state[1][0])    # Generate the key, which is target position.
        measurements_in_gating_area_array = t.read_measurements_within_gating() # Read out the measurements within gating area of current target.
        measurements_in_gating_area = [(x[0][0], x[1][0]) for x in measurements_in_gating_area_array] # Convert it into an array of sets.
        probability_measurement_associated_with_this_target = t.read_probability_for_each_measurement() # Read the likelihood probability of each measurement within the gating area of current target.
        probability_vector = np.zeros(len(unique_measurements)) # Initiate the probability vector, which has the length of len(unique_measurements)
        for measurement in measurements_in_gating_area:     # For every measurement in the gating region of current target:
            idx = unique_measurements.index(measurement) # Get the index of this measurement along the probability vector
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


def compute_marginal_probability(targets, feasible_association_matrices_of_joint_events, joint_probabilities, unique_measurements): # for detailed description please refer to Eq. 3.19 of [2]
    '''
    Compute marginal probability based on feasible association matrix of this measurement and the joint probability as indicated by Eq. 3.19 in [2].
    The marginal association probability of measurement j is associated with target t is the summation all the joint probabilities(joint events) which inlcude this association.

    Input arguments:
        targets: a list of targets
        feasible_association_matrices_of_joint_events: a list of matrices each indicating a joint event
        joint_probabilities: a dictionary for joint probabilities
        unique_measurements: all the measurements associated with targets
    Output:
        marginal_probability with (t_position, m_position) pair as key and the marginal probability between this measurement and this target as value.
    ''' 
    marginal_probability = {} # Initiate Marginal Probability to be an empty dictionary.
    num_matrices = len(feasible_association_matrices_of_joint_events)       # The number of feasible joint events at current frame.
    t_index = 0
    for t in targets:
        t_index += 1 # starting from position one since t_0 is clutter
        t_position = (t.target_state[0][0],t.target_state[1][0])
        for m in t.read_measurements_within_gating():  # Loop over all measurements in the gating region of current target:
            beta = 0  # Initiate Beta to be zero.
            m_position = (m[0][0], m[1][0])
            measurement_idx = unique_measurements.index(m_position) 
            for joint_event_index in range(num_matrices):   # Loop over all feasible association events, for current measurement and current target:
                # Obtain the association matrix for current joint event.
                matrix_of_joint_events = feasible_association_matrices_of_joint_events[joint_event_index]
                if matrix_of_joint_events[measurement_idx][t_index] == 1: # If the association matrix for current joint event indicate that this measurement is associated with this target:
                    # The probability of this association is added to the marginal probability of this measurement. (Since we need to sum over all joint probabilities whose corresponding 
                    # element represents "association between current measurement and current target" in the current association matrix is 1. See equation (3.19) in [2].)
                    beta += joint_probabilities[joint_event_index]
            marginal_probability[(t_position, m_position)] = beta # Associate key to the value.
    return marginal_probability
