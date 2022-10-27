"""
%% ------------------------------ Gaussian Mixture(GM) Cardinalized Probability Hypothesis Density(CPHD) filter ------------------------------ %%
This Python code is reproduction for the "point target GM-CPHD filter" originally proposed in paper [2], with assumption
of no target spawning. The code is adapted from "GM-CPHD" matlab code available at: https://github.com/Agarciafernandez/MTT.

The key idea behind GM-PHD is that when the SNR is large enough, the first statistical moment(PHD) of RFS multi-target density
is used to approximate the multi-target density during recursive propogation process.

As described in Secion II of [2], a major weakness of GM-PHD process is how the states are extracted. When the cardinality 
informaition is unavailable, all we can do is to set a threshold for weights such that the Gaussian components(single target
probability density) with weights below that threshold is ignored. 

The CPHD filter steers a middle ground between the information loss of first-order multitarget-moment approximation and the 
intractability of a full second-order approximation. Cardinalized PHD try to solve it by propogating a cardinality distribution 
along side the first statistical moment of multi-target density.
%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
Processing
% [2] 2006 B.-N. Vo, Anthonio Cantoni "The Cardinalized Probability Hypothesis Density Filter for Linear Gaussian Multi-Target Models", 
2006 Annual Conference on Information Sciences and Systems (CISS)
% [3] 2007 R. Mahler "PHD filters of higher order in target number" IEEE Transactions on Aerospace and Electronic Systems
% [4] 2003 R. Mahler "Multitarget Bayes Filtering via First-Order Multitarget Moments" IEEE Transactions on Aerospace and Electronic Systems
"""
"""
@TODO:
    1. Gating part needs to be corrected, it should be implemented for each target.
    2. The performance of CPHD is NOT good enough, and maybe some parts of code still exist some bugs. Fix these.
"""

from GM_CPHD_Filter_Point_Target import GM_CPHD_Filter
from util import parse_args, gen_ground_truth_parameters, gen_filter_model, gen_simulation,filter_plot,plot_gospa
from matplotlib import pyplot as plt
import time
import numpy as np
import gospa as gp
import pickle
import math

def main(args):
    # Initiate data structure for GOSPA record
    gospa_record_all = []
    gospa_localization_record_all =[]
    gospa_missed_record_all=[]
    gospa_false_record_all=[]

    # Generate simulation
    ground_truth_parameters = gen_ground_truth_parameters()
    
    # Initiate filter
    filter_model = gen_filter_model()
    Filter = GM_CPHD_Filter(filter_model, args.Bayesian_filter_config, args.motion_model_type)
    
    for ith_simulation in range(args.number_of_monte_carlo_simulations):        
        # generate data
        Z_k_all, targetStates_all, observations_all, clutter_all = gen_simulation(ground_truth_parameters,args.n_scan,args.simulation_scenario)
        #Initiate data structure for this simulation.
        gospa_record = []
        gospa_localization_record =[]
        gospa_missed_record=[]
        gospa_false_record=[]
        estimatedStates_record = []
        
        if args.plot:
            fig = plt.figure()
            # Interactive module, the figure will be shown automatically in sequence.
            plt.ion()
            plt.show()
        
        # Start the timer for this simulation
        tic = time.process_time()
        
        for i in range(args.n_scan): # Here we execute processing for each scan time(frame)
            # read out data from simulation
            Z_k = Z_k_all[i]
            targetStates = targetStates_all[i]
            observations = observations_all[i]
            clutter = clutter_all[i]

            if i == 0:  # For the fisrt frame, there are only new birth targets rather than surviving targets thus we call seperate function.
                # the initial step need to prime the system towards a larger cardinality
                predictedCardinality = Filter.cardinality_predict_initial_step()
                # the initial step the labmda for weight update is w_birthinit
                predictedIntensity = Filter.intensity_predict_initial_step()
            else:
                predictedCardinality = Filter.cardinality_predict(updatedCardinality) #Eq. 13 of [2]
                predictedIntensity = Filter.intensity_predict(prunedIntensity)
            '''
            STEP 2: Construct components for gating and intensity update
            '''
            constructedComponentsForIntensityUpdate = Filter.construct_components_for_gating_and_intensity_update_step(predictedIntensity)
            '''
            OPTIONAL STEP: Gating
            '''
            Z_k_gated = Filter.gateMeasurements(Z_k,constructedComponentsForIntensityUpdate, use_gating=filter_model['use_gating']) # select only measurements that falls in the gate
            '''
            STEP 3: Compute the upsilon
            ''' 
            upsilon0,upsilon1,upsilon1_all_minus_one = Filter.compute_upsilon(predictedIntensity, constructedComponentsForIntensityUpdate, Z_k_gated) # Eq. 21 of [2]
            '''
            STEP 4: Update 
            '''
            updatedCardinality = Filter.cardinality_update(upsilon0,predictedCardinality) #Eq. 19 of [2]
            updatedIntensity = Filter.intensity_update(Z_k_gated, predictedIntensity,predictedCardinality, upsilon0, upsilon1, upsilon1_all_minus_one, constructedComponentsForIntensityUpdate) #Eq. 20 of [2]
            '''
            STEP 5: Pruning, Merging
            '''
            prunedIntensity = Filter.pruneAndMerge(updatedIntensity)
            '''
            STEP 6: Extracting estimated states
            '''
            estimates = Filter.extractStates(prunedIntensity, updatedCardinality)  # Extracting estimates from the pruned intensity this gives better result than extracting them from the updated intensity!
            estimatedStates = estimates['m']
            
            # Store Metrics for Plotting
            gospa,target_to_track_assigments,gospa_localization,gospa_missed,gospa_false = gp.calculate_gospa(targetStates, estimatedStates, c=10.0 , p=2, alpha=2)
            gospa_record.append(gospa)
            if len(target_to_track_assigments)!=0:
                # Normalized locallization error = gospa_localization_error/ number of matched targets
                gospa_localization_record.append(math.sqrt(gospa_localization)/len(target_to_track_assigments))
            else:
                # TODO need to check what happen when assignment is 0
                gospa_localization_record.append(math.sqrt(gospa_localization))
            gospa_missed_record.append(math.sqrt(gospa_missed))
            gospa_false_record.append(math.sqrt(gospa_false))
            
            if args.plot:   
                # Plot ground truth states, actual observations, estiamted states and clutter for current scan time(frame).
                filter_plot(fig,targetStates, observations, estimatedStates, clutter,ground_truth_parameters)
         
        # stop timer for this simulation
        toc = time.process_time()
        if args.plot:
            plt.close(fig)
        # print out the processing time for this simulation
        print("This is the %dth monte carlo simulation, GM-CPHD processing takes %f seconds" %(ith_simulation, (toc - tic)))
        # store data for this simulation
        gospa_record_all.append(gospa_record)
        gospa_localization_record_all.append(gospa_localization_record)
        gospa_missed_record_all.append(gospa_missed_record)
        gospa_false_record_all.append(gospa_false_record)

    # Plot the results
    x = range(args.n_scan) 
    gospa_localization_record_all_average = []
    gospa_record_all_average = []
    gospa_missed_record_all_average = []
    gospa_false_record_all_average = []

    for scan in range(args.n_scan):
        gospa_localization_record_all_average.append(np.sum(np.array(gospa_localization_record_all)[:,scan])/args.number_of_monte_carlo_simulations)
        gospa_record_all_average.append(np.sum(np.array(gospa_record_all)[:,scan])/args.number_of_monte_carlo_simulations)
        gospa_missed_record_all_average.append(np.sum(np.array(gospa_missed_record_all)[:,scan])/args.number_of_monte_carlo_simulations)
        gospa_false_record_all_average.append(np.sum(np.array(gospa_false_record_all)[:,scan])/args.number_of_monte_carlo_simulations)
    
    plot_gospa(args.path_to_save_results,args.scenario, x,gospa_record_all_average,gospa_localization_record_all_average,gospa_missed_record_all_average,gospa_false_record_all_average)

if __name__ == '__main__':
    args = parse_args()
    main(args)