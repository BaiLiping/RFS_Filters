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
    parser.add_argument('--path_to_save_results', default='',type=str, help="path to result folder")
    parser.add_argument('--scenario', default='scenario1/',type=str, help="path to scenario folder")
    parser.add_argument('--Bayesian_filter_config', default='Kalman', type=str, help='Config the Bayesian filter used inside filter')
    parser.add_argument('--motion_model_type', default='Constant Velocity',type=str, help='Config the motion_model_type used inside filter')
    parser.add_argument('--simulation_scenario', default="Intersection", type=str, help='scenario for the simulation')
    parser.add_argument('--number_of_monte_carlo_simulations', default=100,type=int, help='number_of_monte_carlo_simulations')
    parser.add_argument('--n_scan', default=101,type=int, help='number frames per simulation')
    parser.add_argument('--death_counter_kill', default=4, type=int, help='The threshold of counter to trigger the deletion of target')
    parser.add_argument('--birth_counter_born', default=4, type=int, help='The threshold of counter to trigger "potential target becomes track')
    parser.add_argument('--birth_initiation', default=2, type=int, help='the initial state of birth counter')
    # choose display configuration
    # silent mode = True : does not display the figure, only print out pertinent information
    # silent mode = False: display figure and print out information
    parser.add_argument('--plot', default=True, type=bool, help='choose if plot')
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
    average_number_of_clutter_per_frame = 100
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
    filter_model = {}  # filter_model is the dictionary which has all the corresponding parameters of the generated filter_model

    T = 1.0  # Sampling period, time step duration between two scans(frames).
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
    sigma_v = 0.1     # Standard deviation of the process noise.
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    filter_model['Q'] = sigma_v ** 2 * Q  # Covariance of process noise

    # Observation/Measurement filter_model parameters (noisy x and y only rather than v_x, v_y):
    filter_model['H'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation filter_model matrix.
    sigma_r = 1
    filter_model['R'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    
    # Measurements parameters. See equation (20) in [1].
    filter_model['P_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D)

    # Compute clutter intensity lambda_c. See equation (47) and corresponding explanation in [1]. 
    average_number_of_clutter_per_frame = 10
    filter_model['average_number_of_clutter_per_frame']=average_number_of_clutter_per_frame
    x_range = [0, 300]  # X range of measurements
    y_range = [0, 300]  # Y range of measurements
    # Compute and store Area
    filter_model['area'] = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area.
    clutterIntensity = average_number_of_clutter_per_frame/filter_model['area']  # Generate clutter intensity (clutter intensity lambda_c = lambda_t/A)
    
    filter_model['clutterIntensity'] = clutterIntensity
    filter_model['unitclutterIntensity'] = 1/filter_model['area']

    # gating threshold
    filter_model['eta'] = 4
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

    # Store Data
    path =  path_to_save_results + '../compare/' + scenario + 'jpda/gospa_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_record_all_average, f)
    f.close()

    path =  path_to_save_results + '../compare/'+scenario+'jpda/gospa_localization_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_localization_record_all_average, f)
    f.close()

    path =  path_to_save_results + '../compare/'+scenario+'jpda/gospa_missed_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_missed_record_all_average, f)
    f.close()

    path =  path_to_save_results + '../compare/'+scenario+'jpda/gospa_false_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_false_record_all_average, f)
    f.close()

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

"For one constant velocity target, perform PDA once per frame."
class Target:
    def __init__(self,
                 x: np.float,
                 y: np.float,
                 F: np.array,
                 H: np.array,
                 Q: np.array,
                 R: np.array,
                 eta: np.float,
                 P_D: np.float,
                 lambda_clutter: np.float,
                 birth_initiation: np.int):
        # lambda_clutter is spatial density of clutter under Poisson clutter model(Thus in this code we use parametric PDA, see equation (37) in [1].).
        self.lambda_clutter = lambda_clutter    # self.lambda_clutter is spatial density of false measurements(clutter).
        self.target_state = np.array([
            [x],
            [y],
            [0],
            [0],
        ])                                     # self.target_state represents the state of each target.
        self.F = F                                          # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
        self.H = H                                          # H is measurement matrix which represents the measurement model in update step.
        self.Q = Q                                          # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R                                          # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.P = np.eye(4, 4)                               # State covariance matrix, P, is initialized using 4-by-4 diagonal matrix for the current target
        self.S = self.innov_cov(self.H, self.P, self.R)     # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        self.eta = eta
        
        self.P_D = P_D  # P_D is the target detection probability.
        self.birth_counter = birth_initiation   # when the target is initiated, counter is set at one                                 
        self.death_counter = 0   # initiate a death counter
        self.measurements_within_gating = []
        self.likelihood_probability_for_each_measurement = {} # a dictionary where the key is the position of each measurement and value its likelihood as seen from this target

    @staticmethod
    def innov_cov(H, P, R):
        # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        return H@P@np.transpose(H) + R

    # Kalman prediction. 
    def predict(self):
        # Calculate all the variables regarding motion model.
        self.target_state = self.F@self.target_state            # Equation (30) in [2].
        self.P = self.F@self.P@np.transpose(self.F) + self.Q    # Equation (32) in [2].
        self.S = self.innov_cov(self.H, self.P, self.R)         # Equation (33) in [2].

    # Kalman update with JPDA.
    def jpda_update(self, marginal_probability, Z: list):
        '''self.S = self.innov_cov(self.H, self.P, self.R) # update self.S based on the new self.P. This step has been accomplished by calling predict. Redundant'''
        zpred = self.zpred() # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = [] # innovation is defined by equation (2.5) in [2]
        betas = []
        t_position = (self.target_state[0][0], self.target_state[1][0])     # Position of current target.
        for z in Z: 
            if z.size != 2:
                raise Exception("z has wrong dimension", z)  # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
            innovations.append(z - zpred) # See eqation (2.14) in [2]
            m_position = (z[0][0], z[1][0])     # Position of current measurement which is in the gating region of current target.
            # Obtain the marginalized association probability between target t and measurement j (in the gating area of target t), beta_j^t with j = 0, 1, 2, ...
            betas.append(marginal_probability[t_position, m_position])
         
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association".
        # According to equation. 3.20 in [2].
        beta_0 = 1 - np.sum(betas)

        # Reduce the mixture into a combined innovation weighted by betas(association probabilities).
        combined_innovation = np.zeros_like(zpred)
        for j, innovation_for_each_measurement in enumerate(innovations):
            combined_innovation += betas[j] * innovation_for_each_measurement               # Calculate the combined innovation, v(k). See equation (40) in [1].
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)   # Calculate the Kalman gain, W(k). See equation (41) in [1].
        self.target_state = self.target_state + W@combined_innovation            # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].

        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j]*innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation@combined_innovation.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov
        print(self.P)

    def pda_update(self, Z: list):
        """ self.S = self.innov_cov(self.H, self.P, self.R)         # I don't understand why we need it again here! """
        zpred = self.zpred()        # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = []
        betas_unorm = []
        for z in Z:
            if z.size != 2:
                # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
                raise Exception("z has wrong dimension", z)
            # Calculate innovation for each measuremnt, (z - zpred), which is v_i(k) in equation (40) in [1].
            innovations.append(z - zpred)
            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) in [1].
            betas_unorm.append(self.P_D * gauss(z, zpred, self.S)/self.lambda_clutter)
            # Potentially a different implementation
            #betas_unorm.append(self.P_D * (gauss(z, zpred, self.S)+np.float64(self.lambda_clutter)))
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

        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j] * \
                innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation @
                        combined_innovation.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov
        print(self.P)

    def zpred(self):
        # Predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        zpred = self.H@self.target_state
        #zpred = np.array([[zpred[0][0]], [zpred[1][0]]])
        return zpred
          
    def gating(self, Z):
        zpred = self.zpred()  # Calculate predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        # Only keep the measurements within the gating area for currect target, according to equation (34) in [1].
        gated_measurements = np.array([z for z in Z if np.transpose(z - zpred)@np.linalg.inv(self.S)@(z - zpred) < self.eta])
        self.measurements_within_gating = gated_measurements
        return gated_measurements.reshape((gated_measurements.size // 2, 2, 1))

    def compute_likelihood_probability_for_each_measurement(self, measurements):
        '''
        Generate a dictionary whose key is the measurement and the corresponding value is its likelihood, See equation (38) or (48) in [1].
        Seeing from this particular target return a dictionary [measurement]: likelihood.
        '''
        for m in measurements:
            if m.size != 2:
                raise Exception("m has wrong dimension", m)
            mpred = self.zpred()
            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) or (48) in [1].
            self.likelihood_probability_for_each_measurement[(m[0][0], m[1][0])] = self.P_D * gauss(m, mpred, self.S)

    def increase_birth_counter(self): # when a potential track has measurement associated, update counter
        self.birth_counter += 1
    def decrease_birth_counter(self):
        self.birth_counter -= 1
    def read_birth_counter(self): # return the birth counter
        return self.birth_counter
    def increase_death_counter(self): # when a potential death track lacks data association, update death counter
        self.death_counter +=1
    def decrease_death_counter(self):
        self.death_counter -=1
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
    F = np.empty((4, 4))
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
    Q = np.eye(4, 4)
    R = np.eye(2, 2)

    def __init__(self, T: np.float, Q: np.array, R: np.array, eta: np.float, P_D: np.float, lambda_clutter: np.float, birth_initiation: np.int):
        self.lambda_clutter = lambda_clutter
        self.Q = Q              # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R              # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])                      # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in the prediction step.
        self.eta = eta
        self.P_D = P_D          # P_D is the target detection probability.
        self.birth_initiation = birth_initiation

    def new(self, x: np.float, y: np.float):
        return Target(x, y, self.F, self.H, self.Q, self.R, self.eta, self.P_D, self.lambda_clutter, self.birth_initiation)
