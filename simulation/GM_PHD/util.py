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
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time
import pickle
import argparse


"""
Parse Arguments
"""
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--path_to_save_results', default='D:/Tech_Resource/Paper_Resource/Signal Processing in General/RFS Filter/PHD Filter/GM_PHD_PointTarget_Python_Demo/',type=str, help="path to result folder")
    parser.add_argument('--path_to_save_results', default='/gs/home/zhubing/Radar_Perception_Project/Project_3/',type=str, help="path to result folder")
    parser.add_argument('--scenario', default='scenario1/',type=str, help="path to scenario folder")
    parser.add_argument('--Bayesian_filter_config', default='Kalman', type=str, help='Config the Bayesian filter used inside filter')
    parser.add_argument('--motion_model_type', default='Constant Velocity',type=str, help='Config the motion_model_type used inside filter')
    parser.add_argument('--simulation_scenario', default="No Intersection Varying Cardinality", type=str, help='scenario for the simulation')
    parser.add_argument('--number_of_monte_carlo_simulations', default=100,type=int, help='number_of_monte_carlo_simulations')
    parser.add_argument('--n_scan', default=101,type=int, help='number frames per simulation')
    # choose display configuration
    # silent mode = True : does not display the figure, only print out pertinent information
    # silent mode = False: display figure and print out information
    parser.add_argument('--plot', default=False, type=bool, help='choose if plot')
    return parser.parse_args()

"""
Ultility functions
"""
def gen_ground_truth_parameters():
    """
    This is the configuration file for all parameters used in GM-PHD filter model, which is used to tracking the multi-targets.
    """
    ground_truth_parameters = {}  # model is the dictionary which has all the corresponding parameters of the generated model

    T = 1.0  # Sampling period, time step duration between two scans(frames).

    # Dynamic motion model parameters(The motion used here is Constant Velocity (CV) model):
    # State transition matrix, F_k.
    # F_k = np.array([
    #         [1, 0, T_s,   0],
    #         [0, 1,   0, T_s],
    #         [0, 0,   1,   0],
    #         [0, 0,   0,   1],
    #     ])
    ground_truth_parameters['F_k'] = np.eye(4, dtype=np.float64)
    I = T*np.eye(2, dtype=np.float64)
    ground_truth_parameters['F_k'][0:2, 2:4] = I
    sigma_v = 0.1     # Standard deviation of the process noise.
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    ground_truth_parameters['Q_k'] = sigma_v ** 2 * Q  # Covariance of process noise
    
    # Initial state covariance matrix, P_k.
    P_k = np.diag([150**2,150**2, 1**2,  1**2])
    ground_truth_parameters['P_k'] = np.array(P_k, dtype=np.float64)

    # Observation/Measurement model parameters (noisy x and y only rather than v_x, v_y):
    ground_truth_parameters['H_k'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation model matrix.
    sigma_r = 1    # Standard deviation of the measurement noise.
    ground_truth_parameters['R_k'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    
    # Measurements parameters. See equation (20) in [1].
    ground_truth_parameters['p_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D)
    
    # Compute clutter intensity. See equation (47) and corresponding explanation in [1]. 
    average_number_of_clutter_per_frame = 10
    x_range = [0, 300]  # X range of measurements
    y_range = [0, 300]  # Y range of measurements
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area.
    clutterIntensity = average_number_of_clutter_per_frame/A  # Generate clutter intensity (clutter intensity lambda_c = lambda_t/A)
    ground_truth_parameters['clutterIntensity'] = clutterIntensity
    ground_truth_parameters['xrange'] = x_range
    ground_truth_parameters['yrange'] = y_range
    ground_truth_parameters['average_number_of_clutter_per_frame']= average_number_of_clutter_per_frame

    return ground_truth_parameters

def gen_filter_model():
    """
    This is the configuration file for all parameters used in GM-PHD filter model, which is used to tracking the multi-targets.
    """
    filter_model = {}  # model is the dictionary which has all the corresponding parameters of the generated model

    T = 1.0  # Sampling period, time step duration between two scans(frames).

    # Dynamic motion model parameters(The motion used here is Constant Velocity (CV) model):
    # State transition matrix, F_k.
    # F_k = np.array([
    #         [1, 0, T_s,   0],
    #         [0, 1,   0, T_s],
    #         [0, 0,   1,   0],
    #         [0, 0,   0,   1],
    #     ])
    filter_model['F_k'] = np.eye(4, dtype=np.float64)
    I = T*np.eye(2, dtype=np.float64)
    filter_model['F_k'][0:2, 2:4] = I
    sigma_v = 0.1     # Standard deviation of the process noise.
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    filter_model['Q_k'] = sigma_v ** 2 * Q  # Covariance of process noise
    
    # Initial state covariance matrix, P_k.
    P_k = np.diag([150**2,150**2, 1**2,  1**2])
    filter_model['P_k'] = np.array(P_k, dtype=np.float64)

    # Observation/Measurement model parameters (noisy x and y only rather than v_x, v_y):
    filter_model['H_k'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation model matrix.
    sigma_r = 1    # Standard deviation of the measurement noise.
    filter_model['R_k'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    
    # Measurements parameters. See equation (20) in [1].
    filter_model['p_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D)
    filter_model['p_S'] = 0.99

    # Compute clutter intensity. See equation (47) and corresponding explanation in [1]. 
    average_number_of_clutter_per_frame = 10
    x_range = [0, 300]  # X range of measurements
    y_range = [0, 300]  # Y range of measurements
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area.
    clutterIntensity = average_number_of_clutter_per_frame/A  # Generate clutter intensity (clutter intensity lambda_c = lambda_t/A)
    filter_model['clutterIntensity'] = clutterIntensity
    filter_model['xrange'] = x_range
    filter_model['yrange'] = y_range
    filter_model['average_number_of_clutter_per_frame']= average_number_of_clutter_per_frame

    # Define gating threshold
    filter_model['use_gating'] = False
    filter_model['gating_threshold'] = 10
    # cap the number of Gaussian components
    filter_model['capping_gaussian_components'] = True
    filter_model['maximum_number_of_gaussian_components'] = 100
    # choose if cholsky is used when compute invS
    filter_model['using_cholsky_decomposition_for_calculating_inverse_of_measurement_covariance_matrix'] = False

    # GM-PHD filter merge, pruning and state extraction parameters, see tabel II in [1].
    filter_model['T'] = 10**-4  # Pruning weight threshold.
    filter_model['U'] = 0.2  # Merge distance threshold.
    filter_model['w_thresh'] = 0.5 # State extraction weight threshold(i.e. Existence probability used to extract estimates, only the Gaussian components with weight higher than this threshold will be extracted)

    # Measurements parameters. See equation (20) in [1].
    filter_model['p_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D)
    # Survival/death of targets.  See equation (19) in [1].
    filter_model['p_S'] = 0.99 # Probability of target survival (prob_death = 1 - prob_survival)

    # Target birth parameters
    filter_model['w_birthsum'] = 0.005 #0.02 # The total weight of birth targets. It is chosen depending on handling false positives.
    filter_model['n_birth'] = 1

    # Target Birth Initial Step
    filter_model['w_birthsuminit'] = 1 #0.02 # The total weight of birth targets. It is chosen depending on handling false positives.
    filter_model['n_birthinit'] = 4
    filter_model['z_init'] = [[100+50*x, 100-50*x] for x in range(filter_model['n_birthinit'])]

    return filter_model

def gen_ground_truth_states(ground_truth_parameters, targetStates, noiseless):
    """
    Generate ground truth states of all targets per scan
    """
    truthStates = []
    for i in range(len(targetStates)):
        if noiseless == True:
            # New target ground truth state without any changes on the initial state setting up, which means the moving trajectary will
            # be a straight line with constant velocity.
            truthState = ground_truth_parameters['F_k'].dot(targetStates[i])
            truthStates.append(truthState)
        else:
            # New target ground truth state with changes('process noise') on the initial state setting up, which means the moving trajectary
            # will not be a straight line with constant velocity but can be any shape and velocity.
            W = np.sqrt(ground_truth_parameters['Q_k']).dot(np.random.randn(ground_truth_parameters['Q_k'].shape[0], targetStates[i].shape[1])) #.reshape(-1,1) # Process noise
            truthState = ground_truth_parameters['F_k'].dot(targetStates[i]) + W   # New target true state
            truthStates.append(truthState)
    return truthStates

def gen_observations(ground_truth_parameters, truthStates):
    """
    Generate observations of targets per scan
    """
    obserations = []
    # We are not guaranteed to detect the target - there is only a probability
    for i in range(len(truthStates)):
        detect_i = np.random.rand() # Uniformly distribution.
        if detect_i <= ground_truth_parameters['p_D']:    # Only generate observations for those targets which have been detected
            V = np.sqrt(ground_truth_parameters['R_k']).dot(np.random.randn(ground_truth_parameters['R_k'].shape[0], truthStates[i].shape[1]) ) #.reshape(-1,1) # Observation noise
            # Beware there is only one observation which comes from the target which has been detected, that is why this is point target model
            observation = ground_truth_parameters['H_k'].dot(truthStates[i]) + V
            obserations.append(observation)
    return obserations

def gen_clutter(ground_truth_parameters):
    """
    Generate clutter of whole area per scan
    """
    number_of_clutter_in_current_frame = np.random.poisson(ground_truth_parameters['average_number_of_clutter_per_frame'])
    x_range = ground_truth_parameters['xrange']
    y_range = ground_truth_parameters['yrange']
    clutter = []
    for n in range(number_of_clutter_in_current_frame):
        clutterX =  np.random.rand() * (x_range[1] - x_range[0]) + x_range[0] # Random number between x_range[0] and x_range[1], uniformly distributed.
        clutterY = np.random.rand() * (y_range[1] - y_range[0]) + y_range[0]
        clutter.append(np.array([clutterX, clutterY]).reshape(-1,1))
    return clutter

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

# generate simulation
def gen_simulation(ground_truth_parameters,n_scan, simulation_scenario):
    """
    scenario:
    No Intersection
    Intersection
    No Intersection Varying Cardinality
    Intersection Varying Cardinality
    """
    # initiate the data structure to be a list of n_scan dictionaries
    Z_k_all = [{} for i in range(n_scan)]
    targetStates_all = [{} for i in range(n_scan)]
    observations_all = [{} for i in range(n_scan)]
    clutter_all = [{} for i in range(n_scan)]

    if simulation_scenario == "No Intersection":
        # Setup the starting target states (positions , velocities)
        m_start1 = np.array([230, 220, -1.3, -0.5]).reshape(4, 1)
        m_start2 = np.array([250,250, -0.5, -1.3]).reshape(4, 1)
        m_start3 = np.array([100, 100, 0.7, 0.5]).reshape(4, 1)
        m_start4 = np.array([100, 250, 0.5, -0.8]).reshape(4, 1)
    
        # Initialize the initial position of targets
        targetStates_init = [m_start1, m_start2, m_start3, m_start4]

        for i in range(n_scan):
            if i == 0:
                targetStates_i = targetStates_init
            # generate data for each frame
            targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
            observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
            clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
            Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
            
            # store data
            Z_k_all[i]=Z_k_i
            targetStates_all[i]=targetStates_i
            observations_all[i]=observations_i
            clutter_all[i]=clutter_i
    
    elif simulation_scenario == "Intersection":
        # Setup the n_scan/2 target states (positions , velocities)
        m_start1 = np.array([150, 150, -1.3, -0.5]).reshape(4, 1)
        m_start2 = np.array([150, 150, -0.5, -1.3]).reshape(4, 1)
        m_start3 = np.array([150, 150, 0.7, 0.5]).reshape(4, 1)
        m_start4 = np.array([150, 150, 0.5, -0.8]).reshape(4, 1)
    
        # Initialize the initial position of targets at n_scan/2
        targetStates = [m_start1, m_start2, m_start3, m_start4]
        
        if n_scan%2 == 0:
            IOError("Please make sure that n_scan is an odd number")
        else:
            for i in range(int((n_scan-1)/2)+1)[1:]:
                if i == 1:
                    targetStates_i = targetStates
                # generate data from frame 0 to fram n_scan/2 -1
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)-i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)-i]=targetStates_i
                observations_all[int((n_scan-1)/2)-i]=observations_i
                clutter_all[int((n_scan-1)/2)-i]=clutter_i
            
            for i in range(int((n_scan-1)/2)+1):
                if i == 0:
                    targetStates_i = targetStates
                # generate data from frame n_scan/2 to n_scan
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)+i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)+i]=targetStates_i
                observations_all[int((n_scan-1)/2)+i]=observations_i
                clutter_all[int((n_scan-1)/2)+i]=clutter_i

    elif simulation_scenario == "No Intersection Varying Cardinality":
        # Setup the starting target states (positions , velocities)
        m_start1 = np.array([230, 220, -1.3, -0.5]).reshape(4, 1)
        m_start2 = np.array([250,250, -0.5, -1.3]).reshape(4, 1)
        m_start3 = np.array([100, 100, 0.7, 0.5]).reshape(4, 1)
        m_start4 = np.array([100, 250, 0.5, -0.8]).reshape(4, 1)
    
        # Initialize the initial position of targets
        targetStates = [m_start1, m_start2, m_start3, m_start4]

        for i in range(n_scan):
            if i == 0:
                targetStates_i = targetStates
            # generate data for each frame
            targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
            # every 10 frames
            if i/30 == 1:
                targetStates_i.remove(targetStates_i[0])
            if i/30 ==2:
                new_target = np.array([10,10,1,1]).reshape(4, 1)
                targetStates_i.append(new_target)
            if i/30 == 3:
                targetStates_i.remove(targetStates_i[0])
            if i/30 == 4:
                new_target = np.array([30,30,1,1]).reshape(4, 1)
                targetStates_i.append(new_target)
            if i/30 == 5:
                targetStates_i.remove(targetStates_i[0])
            if i/30 ==6:
                new_target = np.array([50,50,1,1]).reshape(4, 1)
                targetStates_i.append(new_target)
            if i/30 == 7:
                targetStates_i.remove(targetStates_i[0])
            if i/30 == 8:
                new_target = np.array([70,70,1,1]).reshape(4, 1)
                targetStates_i.append(new_target)
            if i/30 == 9:
                new_target = np.array([90,90,1,1]).reshape(4, 1)
                targetStates_i.append(new_target)
            
            #if i%10 == 0 and i != 0:
                # either a target disapear or appear
            #    choice = np.random.choice([0,1])
                # if the choice is 0 and there are at least 2 targets, then disappear
            #    if choice ==  0 and len(targetStates_i)>1:
            #        targetStates_i.remove(targetStates_i[0])
            #    else:
            #        position = np.random.uniform(20,280,2)
            #        velocity = np.random.uniform(-1,1,2)
            #        new_target = np.array([position[0],position[1], velocity[0],velocity[1]]).reshape(4, 1)
            #        targetStates_i.append(new_target)
                    
            observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
            clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
            Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
            
            # store data
            Z_k_all[i]=Z_k_i
            targetStates_all[i]=targetStates_i
            observations_all[i]=observations_i
            clutter_all[i]=clutter_i
    
    elif simulation_scenario == "Intersection Varying Cardinality":
        # Setup the n_scan/2 target states (positions , velocities)
        m_start1 = np.array([150, 150, -1.3, -0.5]).reshape(4, 1)
        m_start2 = np.array([150, 150, -0.5, -1.3]).reshape(4, 1)
        m_start3 = np.array([150, 150, 0.7, 0.5]).reshape(4, 1)
        m_start4 = np.array([150, 150, 0.5, -0.8]).reshape(4, 1)
    
        # Initialize the initial position of targets at n_scan/2
        targetStates = [m_start1, m_start2, m_start3, m_start4]
        
        if n_scan%2 == 0:
            IOError("Please make sure that n_scan is an odd number")

        else:
            for i in range(int((n_scan-1)/2)+1)[1:]:
                if i == 1:
                    targetStates_i = targetStates
                # generate data from frame 0 to fram n_scan/2 -1
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                if i/30 == 1:
                    targetStates_i.remove(targetStates_i[0])
                if i/30 ==2:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                if i/30 == 3:
                    targetStates_i.remove(targetStates_i[0])
                if i/30 == 4:
                    new_target = np.array([30,30,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)

                # every 10 frames
                #if i%10 == 0 and i != 0 and i != (int((n_scan-1)/2)):
                    # either a target disapear or appear
                #    choice = np.random.choice([0,1])
                    # if the choice is 0 and there are at least 2 targets, then disappear
                #    if choice ==  0 and len(targetStates_i)>1:
                #        targetStates_i.remove(targetStates_i[0])
                #    else:
                #        position = np.random.uniform(20,280,2)
                #        velocity = np.random.uniform(-1,1,2)
                #        new_target = np.array([position[0],position[1], velocity[0],velocity[1]]).reshape(4, 1)
                #        targetStates_i.append(new_target)
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)-i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)-i]=targetStates_i
                observations_all[int((n_scan-1)/2)-i]=observations_i
                clutter_all[int((n_scan-1)/2)-i]=clutter_i
            
            for i in range(int((n_scan-1)/2)+1):
                if i == 0:
                    targetStates_i = targetStates
                # generate data from frame n_scan/2 to n_scan
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                if i/30 == 1:
                    targetStates_i.remove(targetStates_i[0])
                if i/30 ==2:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                if i/30 == 3:
                    targetStates_i.remove(targetStates_i[0])
                if i/30 == 4:
                    new_target = np.array([30,30,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)

                # every 10 frames
                #if i%10 == 0 and i != 0 and i != (int((n_scan-1)/2)):
                #    # either a target disapear or appear
                #    choice = np.random.choice([0,1])
                #    # if the choice is 0 and there are at least 2 targets, then disappear
                #    if choice ==  0 and len(targetStates_i)>1:
                #        targetStates_i.remove(targetStates_i[0])
                #    else:
                #        position = np.random.uniform(20,280,2)
                #        velocity = np.random.uniform(-1,1,2)
                #        new_target = np.array([position[0],position[1], velocity[0],velocity[1]]).reshape(4, 1)
                #        targetStates_i.append(new_target)
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)+i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)+i]=targetStates_i
                observations_all[int((n_scan-1)/2)+i]=observations_i
                clutter_all[int((n_scan-1)/2)+i]=clutter_i
    elif simulation_scenario == "Travel in Proximity":
        m_start1 = np.array([10, 10, 1.3, 1.5]).reshape(4, 1)
        m_start2 = np.array([12, 17, 1.5, 1.3]).reshape(4, 1)
        m_start3 = np.array([8, 15, 1.7, 1.5]).reshape(4, 1)
        m_start4 = np.array([20, 15, 1.5, 1.8]).reshape(4, 1)
    
        # Initialize the initial position of targets
        targetStates = [m_start1, m_start2, m_start3, m_start4]

        for i in range(n_scan):
            if i == 0:
                targetStates_i = targetStates
            # generate data for each frame
            targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
            observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
            clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
            Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
            
            # store data
            Z_k_all[i]=Z_k_i
            targetStates_all[i]=targetStates_i
            observations_all[i]=observations_i
            clutter_all[i]=clutter_i
    elif simulation_scenario == "Intersection More than one Cardinality Changes":
        # Setup the n_scan/2 target states (positions , velocities)
        m_start1 = np.array([155, 155, -1.3, -0.5]).reshape(4, 1)
        m_start2 = np.array([150, 150, -0.5, -1.3]).reshape(4, 1)
        m_start3 = np.array([140, 140, 0.7, 0.5]).reshape(4, 1)
        m_start4 = np.array([145, 145, 0.5, -0.8]).reshape(4, 1)
    
        # Initialize the initial position of targets at n_scan/2
        targetStates = [m_start1, m_start2, m_start3, m_start4]
        
        if n_scan%2 == 0:
            IOError("Please make sure that n_scan is an odd number")

        else:
            for i in range(int((n_scan-1)/2)+1)[1:]:
                if i == 1:
                    targetStates_i = targetStates
                # generate data from frame 0 to fram n_scan/2 -1
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                if i/20 == 1:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([20,50,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])
                if i/20 ==2:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([20,50,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])
                if i/20 == 3:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([20,50,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])
                if i/20 == 4:
                    new_target = np.array([10,10,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([20,20,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])

                # every 10 frames
                #if i%10 == 0 and i != 0 and i != (int((n_scan-1)/2)):
                    # either a target disapear or appear
                #    choice = np.random.choice([0,1])
                    # if the choice is 0 and there are at least 2 targets, then disappear
                #    if choice ==  0 and len(targetStates_i)>1:
                #        targetStates_i.remove(targetStates_i[0])
                #    else:
                #        position = np.random.uniform(20,280,2)
                #        velocity = np.random.uniform(-1,1,2)
                #        new_target = np.array([position[0],position[1], velocity[0],velocity[1]]).reshape(4, 1)
                #        targetStates_i.append(new_target)
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)-i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)-i]=targetStates_i
                observations_all[int((n_scan-1)/2)-i]=observations_i
                clutter_all[int((n_scan-1)/2)-i]=clutter_i
            
            for i in range(int((n_scan-1)/2)+1):
                if i == 0:
                    targetStates_i = targetStates
                # generate data from frame n_scan/2 to n_scan
                targetStates_i = gen_ground_truth_states(ground_truth_parameters, targetStates_i, noiseless = False) #add guassian noise to desired targetStates
                if i/20 == 1:
                    new_target = np.array([250,250,-1,-1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([200,250,-1,-1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])
                if i/20 ==2:
                    new_target = np.array([250,250,-1,-1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    new_target = np.array([200,250,-1,-1]).reshape(4, 1)
                    targetStates_i.append(new_target)
                    targetStates_i.remove(targetStates_i[0])
                    targetStates_i.remove(targetStates_i[0])
                if i/20 == 3:
                    targetStates_i.remove(targetStates_i[0])
                if i/20 == 4:
                    new_target = np.array([30,30,1,1]).reshape(4, 1)
                    targetStates_i.append(new_target)

                # every 10 frames
                #if i%10 == 0 and i != 0 and i != (int((n_scan-1)/2)):
                #    # either a target disapear or appear
                #    choice = np.random.choice([0,1])
                #    # if the choice is 0 and there are at least 2 targets, then disappear
                #    if choice ==  0 and len(targetStates_i)>1:
                #        targetStates_i.remove(targetStates_i[0])
                #    else:
                #        position = np.random.uniform(20,280,2)
                #        velocity = np.random.uniform(-1,1,2)
                #        new_target = np.array([position[0],position[1], velocity[0],velocity[1]]).reshape(4, 1)
                #        targetStates_i.append(new_target)
                observations_i = gen_observations(ground_truth_parameters, targetStates_i) #add measurement noise to observation
                clutter_i = gen_clutter(ground_truth_parameters) #get the clutter generated by this environment
                Z_k_i = observations_i + clutter_i # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
             
                # store data
                Z_k_all[int(n_scan/2)+i]=Z_k_i
                targetStates_all[int((n_scan-1)/2)+i]=targetStates_i
                observations_all[int((n_scan-1)/2)+i]=observations_i
                clutter_all[int((n_scan-1)/2)+i]=clutter_i
      
    else:
        print('you have entered {} as simulation scenario'.format(simulation_scenario))
        print('please enter the valid simulation scenario')
    return Z_k_all, targetStates_all, observations_all, clutter_all

# plot function for the demo
def filter_plot(fig,truthStates, observations, estimatedStates, clutter,ground_truth_parameters):
    """
    Plot all information(ground truth states, measurements which include both observations and clutters, 
    and estimated states for all targets) per scan(frame)
    """
    # Plot the ground truth state of targets.
    for i in range(len(truthStates)):
        truthState = truthStates[i]
        # plt.plot(truthState[0], truthState[1], '.b', markersize = 10.0, label='ground truth')
        plt.plot(truthState[0], truthState[1], '.b', markersize = 10.0)

    # Plot the measurements.
    for i in range(len(observations)):
        observation = observations[i]
        # plt.plot(observation[0], observation[1], '.r', markersize = 10.0, label='measurement')
        if len(observation) > 0:
            plt.plot(observation[0], observation[1], '.r', markersize = 10.0)

    # Plot the clutters.
    for i in range(len(clutter)):
        clut = clutter[i]
        # plt.plot(observation[0], observation[1], 'xk', markersize = 5.0, label='clutter')
        plt.plot(clut[0], clut[1], 'xk', markersize = 5.0)

    # Plot the estimated state of targets.
    for i in range(len(estimatedStates)):
        estimatedState = np.array(estimatedStates[i], dtype=np.float64)
        # plt.plot(estimatedState[0], estimatedState[1], '.g', markersize = 10.0, label='estimated state')
        plt.plot(estimatedState[0], estimatedState[1], '.g', markersize = 10.0)

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Ground truth (blue), observations (red), estimated states (green) and clutter (black x)', fontsize=8)
    plt.xlim((ground_truth_parameters['xrange'][0], ground_truth_parameters['xrange'][1]))
    plt.ylim((ground_truth_parameters['yrange'][0], ground_truth_parameters['yrange'][1]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

def plot_gospa(path_to_save_results,scenario,x,gospa_record_all_average,gospa_localization_record_all_average,gospa_missed_record_all_average,gospa_false_record_all_average):
    '''
    plt.title("RMS GOSPA Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Error") 
    plt.plot(x,gospa_record_all_average) 
    #plt.show()
    plt.savefig(path_to_save_results +scenario+'gospa_error.png')
    plt.close()

    plt.title("RMS GOSPA Normalized Localization Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Normalized Localization Error") 
    plt.plot(x,gospa_localization_record_all_average) 
    #plt.show()
    plt.savefig(path_to_save_results +scenario+'localization.png')
    plt.close()

    plt.title("RMS GOSPA Missed Target Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Missed Target Error") 
    plt.plot(x,gospa_missed_record_all_average) 
    #plt.show()
    plt.savefig(path_to_save_results + scenario+'missed.png')
    plt.close()

    plt.title("RMS GOSPA False Target Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA False Target Error") 
    plt.plot(x,gospa_false_record_all_average) 
    #plt.show()
    plt.savefig(path_to_save_results +scenario+'false.png')
    plt.close()
    '''

    # Store Data
    path =  path_to_save_results + 'compare/' + scenario + 'gm_phd/gospa_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'gm_phd/gospa_localization_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_localization_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'gm_phd/gospa_missed_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_missed_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'gm_phd/gospa_false_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_false_record_all_average, f)
    f.close()

