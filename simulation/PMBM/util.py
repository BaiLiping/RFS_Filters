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
One important thing to keep in mind while reading through this document is that the procedures are not
as what is presented in paper [1]. While the peudo code is good for a ballpark sketch of what PMBM tries
to do, the deveils are all in detail, some of which are clearly empirically motivated.  
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
for mis-detection hypothesis and clutter, scoring each permutation, rank them and then propogate the best k permutations to the next
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
        filter_upd['tracks'][i]['weight_of_single_target_hypothesis_in_log_format']:
            a list contains the log weight information of data associations (indices to measurements or 0 if undetected) for the all Gaussian components(all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['weight_of_single_target_hypothesis_in_log_format'][j] is a list which contains the log weight info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
"""
"""
The current version of this code, is updated in 20210902.
"""


"""
Configuration of parameters of the PMBM filter simulation_simulation_model.
"""

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
def gen_simulation_model():
    """
    This is the configuration file for all parameters used in PMBM filter simulation_model, which is used to tracking the multi-targets.
    
        simulation_model['F_k']
        simulation_model['Q_k'] 
        simulation_model['H_k']
        simulation_model['R_k']
        simulation_model['xrange']
        simulation_model['yrange']
        simulation_model['clutter_intensity']
        simulation_model['p_D']
    """
    simulation_model = {}  # simulation_model is the dictionary which has all the corresponding parameters of the generated simulation_model

    T = 1.0  # Sampling period, time step duration between two scans(frames).

    # Dynamic motion simulation_model parameters(The motion used here is Constant Velocity (CV) simulation_model)
    # State transition matrix, F_k.
    # F_k = np.array([
    #         [1, 0, T_s,   0],
    #         [0, 1,   0, T_s],
    #         [0, 0,   1,   0],
    #         [0, 0,   0,   1],
    #     ])
    simulation_model['F_k'] = np.eye(4, dtype=np.float64)
    I = T*np.eye(2, dtype=np.float64)
    simulation_model['F_k'][0:2, 2:4] = I
    # Standard deviation of the process noise.
    sigma_v = 0.1   
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    simulation_model['Q_k'] = sigma_v ** 2 * Q  # Covariance of process noise.

    # Observation/Measurement simulation_model parameters (noisy x and y only rather than v_x, v_y)
    simulation_model['H_k'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation simulation_model matrix.
    sigma_r = 1       # Standard deviation of the measurement noise.
    simulation_model['R_k'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    
    # Measurements parameters
    simulation_model['p_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D).

    # Initial state covariance
    P_k = np.diag([150.**2, 150**2., 1., 1.])
    simulation_model['P_new_birth'] = np.array(P_k, dtype=np.float64)     # Every new target is born with its variance.

    # Clutter(False alarm) parameters
    x_range = [0, 300]  # X range of measurements
    y_range = [0, 300]  # Y range of measurements
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area
    average_number_of_clutter_per_frame = 10
    simulation_model['clutter_intensity'] = average_number_of_clutter_per_frame/A  # Generate clutter/false alarm intensity (clutter intensity lambda_c = lambda_t/A)
    simulation_model['xrange'] = x_range
    simulation_model['yrange'] = y_range
    simulation_model['average_number_of_clutter_per_frame']=average_number_of_clutter_per_frame


    return simulation_model

def gen_filter_model():
    """
    This is the configuration file for all parameters used in PMBM filter simulation_model, which is used to tracking the multi-targets.
    
        Kalman Filter Parameters:
        filter_model['F_k']
        filter_model['Q_k'] 
        filter_model['H_k']
        filter_model['R_k']
        Environment Simulation Parameters
        filter_model['xrange']
        filter_model['yrange']
        filter_model['clutter_intensity']
        filter_model['p_D']
        filter_model['p_S']
        filter_model['x_new_birth']
        filter_model['P_newbirth'] 
        filter_model['number_of_new_birth_targets']
        filter_model['number_of_new_birth_targets_init']
        filter_model['w_birthsum']
        filter_model['w_birthsuminit']
        Filter Parameters 
        filter_model['maximum_number_of_global_hypotheses']
        filter_model['T_pruning_MBM']
        filter_model['T_pruning_Pois']
        filter_model['eB_threshold']
        filter_model['gating_threshold']
        filter_model['state_extraction_option']
        filter_model['eB_estimation_threshold']
    """
    filter_model = {}  # filter_model is the dictionary which has all the corresponding parameters of the generated filter_model

    T = 1.0  # Sampling period, time step duration between two scans(frames).

    # Dynamic motion filter_model parameters(The motion used here is Constant Velocity (CV) filter_model)
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
    # Standard deviation of the process noise.
    sigma_v = 0.1   
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    filter_model['Q_k'] = sigma_v ** 2 * Q  # Covariance of process noise.
    #filter_model['Q_k']=np.diag([0.1,0.1,0.1,0.1])

    # Observation/Measurement filter_model parameters (noisy x and y only rather than v_x, v_y)
    filter_model['H_k'] = np.array([[1., 0, 0, 0], [0, 1., 0, 0]], dtype=np.float64)  # Observation filter_model matrix.
    sigma_r = 1       # Standard deviation of the measurement noise.
    filter_model['R_k'] = sigma_r ** 2 * np.eye(2, dtype=np.float64)  # Covariance of observation noise (change with the size of detection?).
    
    # Measurements parameters
    filter_model['p_D'] = 0.90  # Probability of measurements of targets(The probability target could be detected, so probability of miss-detected of targets is 1 - p_D).
    # Survival/death of targets
    filter_model['p_S'] = 0.99 # Probability of target survival (prob_death = 1 - prob_survival).

    # PMBM filter pruning 
    filter_model['maximum_number_of_global_hypotheses'] = 200     # Maximum number of hypotheses(MBM components)
    filter_model['T_pruning_MBM'] = 1e-4     # Threshold for pruning multi-Bernoulli mixtures weights.
    filter_model['T_pruning_Pois'] = 1e-4   # Threshold for pruning PHD of the Poisson component.
    filter_model['eB_threshold'] = 1e-5
    # Gating parameters
    filter_model['gating_threshold'] = 20

    # Initial state covariance
    P_k = np.diag([150.**2, 150**2., 1., 1.])
    filter_model['P_new_birth'] = np.array(P_k, dtype=np.float64)     # Every new target is born with its variance.


    # Initialize parameters of new Poisson birth target
    filter_model['number_of_new_birth_targets'] = 1    # (Assumed)Number of new birth target.
    filter_model['m_new_birth']=[100,100,1,1] # the fixed postion of birth
    filter_model['w_birthsum'] = 0.005 #0.02 # The total weight of all new birth targets. It is chosen depending on handling false positives.
    
    # Initialize parameters of first step Poisson birth
    filter_model['number_of_new_birth_targets_init'] = 1
    filter_model['w_birthsuminit'] = 4

    # Clutter(False alarm) parameters
    x_range = [0, 300]  # X range of measurements
    y_range = [0, 300]  # Y range of measurements
    A = (x_range[1] - x_range[0])*(y_range[1]-y_range[0])   # Size of area
    average_number_of_clutter_per_frame = 10
    filter_model['clutter_intensity'] = average_number_of_clutter_per_frame/A  # Generate clutter/false alarm intensity (clutter intensity lambda_c = lambda_t/A)
    filter_model['xrange'] = x_range
    filter_model['yrange'] = y_range
    filter_model['average_number_of_clutter_per_frame']=average_number_of_clutter_per_frame

    # Set the option for state extraction step
    filter_model['state_extraction_option'] = 1
    filter_model['eB_estimation_threshold'] = 0.4

    return filter_model

"""
Ultility functions to mimic the scenarios.
"""
def gen_ground_truth_states(simulation_model, targetStates, noiseless):
    """
    Generate ground truth states of all targets per scan
    """
    truthStates = []
    for i in range(len(targetStates)):
        if noiseless == True:
            # New target ground truth state without any changes on the initial state setting up, which means the moving trajectary will
            # be a straight line with constant velocity.
            truthState = simulation_model['F_k'].dot(targetStates[i])
            truthStates.append(truthState)
        else:
            # New target ground truth state with changes('process noise') on the initial state setting up, which means the moving trajectary
            # will not be a straight line with constant velocity but can be any shape and velocity.
            W = np.sqrt(simulation_model['Q_k']).dot(np.random.randn(simulation_model['Q_k'].shape[0], targetStates[i].shape[1])) #.reshape(-1,1) # Process noise
            truthState = simulation_model['F_k'].dot(targetStates[i]) + W   # New target true state
            truthStates.append(truthState)

    return truthStates

def gen_observations(simulation_model, truthStates):
    """
    Generate observations of targets per scan
    """
    obserations = []
    # We are not guaranteed to detect the target - there is only a probability
    for i in range(len(truthStates)):
        detect_i = np.random.rand() # Uniformly distribution.
        if detect_i <= simulation_model['p_D']:    # Only generate observations for those targets which have been detected
            V = np.sqrt(simulation_model['R_k']).dot(np.random.randn(simulation_model['R_k'].shape[0], truthStates[i].shape[1]) ) #.reshape(-1,1) # Observation noise
            # Beware there is only one observation which comes from the target which has been detected, that is why this is point target simulation_model
            observation = simulation_model['H_k'].dot(truthStates[i]) + V
            obserations.append(observation)

    return obserations

def gen_clutter(simulation_model):
    """
    Generate clutter of whole area per scan
    """
    number_of_clutter_in_current_frame = np.random.poisson(simulation_model['average_number_of_clutter_per_frame'])
    x_range = simulation_model['xrange']
    y_range = simulation_model['yrange']
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

# The true targetStates will be obfuscated by the environment
def gen_data_from_environment(simulation_model, targetStates):
        targetStates = gen_ground_truth_states(simulation_model, targetStates, noiseless = False) # Add guassian noise to desired targetStates
        observations = gen_observations(simulation_model, targetStates) # Add measurement noise to observation
        clutter = gen_clutter(simulation_model) # Get the clutter generated by this environment
        Z_k = observations + clutter # Add clutter to the observations to mimic measruements under cluttered environment(The measurements is union of observations and clutter)
        return Z_k, targetStates, observations, clutter # Z_k is the only input of the filter, but noisy_targetStates, observations, clutter are required for plotting purposes

def CardinalityMB(r):
    '''
    Gasia'a original code for this function
    %Two options to compute the cardinality distribution of a multi-Bernoulli
    %RFS, one using FFT and the other one direct convolution
    
    %We calculate the cardinality of a MB distribution using FFT
    N = length(r);
    exp_omega = exp(-1i*(0:N)/(N+1)*2*pi);
    F = ones(1,N+1);
    for i = 1:N
        F = F .* ((1-r(i)) + r(i)*exp_omega);
    end
    pcard = real(ifft(F));
    
    %We calculate the cardinality of a MB distribution using direct convolution
    % N = length(r);
    % pcard = zeros(1,N+1);
    % pcard(1) = 1;
    % for i = 1:N
    %   pcard(2:i+1) = (1-r(i))*pcard(2:i+1) + r(i)*pcard(1:i);
    %   pcard(1) = (1-r(i))*pcard(1);
    % end
    '''
    # Calculate the cardinality of a MB distribution using direct convolution.
    N = len(r)
    pcard = np.zeros(N+1)
    pcard[1] = 1
    for i in range(N):
        pcard[1:] = (1-r[i])*pcard[1:] + r[i]*pcard[0:i-1]
        pcard[0] = (1-r[i])*pcard[0]

    return pcard

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

def groud_truth_plot(fig, truthStates,simulation_model, simulation_scenario):
    # color scheme
    color_scheme =['.b','.g','.r','.m','.y','.c','.k']
    # Plot the ground truth state of targets.
    for i in range(len(truthStates)):
        truthState = truthStates[i]
        # plt.plot(truthState[0], truthState[1], '.b', markersize = 10.0, label='ground truth')
        plt.plot(truthState[0], truthState[1], color_scheme[i], markersize = 10.0)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    #plt.title('Ground truth (blue), observations (red), estimated states (green) and clutter (black x)', fontsize=8)
    plt.title('{} Ground Truth'.format(simulation_scenario))
    plt.xlim((simulation_model['xrange'][0], simulation_model['xrange'][1]))
    plt.ylim((simulation_model['yrange'][0], simulation_model['yrange'][1]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)


# plot function for the demo
def filter_plot(fig,truthStates, observations, estimatedStates, clutter,simulation_model):
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
    plt.xlim((simulation_model['xrange'][0], simulation_model['xrange'][1]))
    plt.ylim((simulation_model['yrange'][0], simulation_model['yrange'][1]))
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
    path =  path_to_save_results + 'compare/' + scenario + 'pmbm/gospa_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'pmbm/gospa_localization_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_localization_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'pmbm/gospa_missed_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_missed_record_all_average, f)
    f.close()

    path =  path_to_save_results + 'compare/'+scenario+'pmbm/gospa_false_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_false_record_all_average, f)
    f.close()