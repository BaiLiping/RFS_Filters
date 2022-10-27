"""
%% ----------------------- Run Gaussian Mixture(GM) Probability Hypothesis Density(PHD) filter ---------------------- %%
This is script which runs "point target GM-PHD filter" originally proposed in paper [1], with assumption
of no target spawning.

%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
Processing
"""
"""
The current version of this code, is updated in 20210810.
"""
"""
@TODO:
    1. Create a new type of GM-PHD filter with usage of global hypothesis based data association, and eventurely marginalize multiple global hypotheses into one(like how the 
        data association and update is handled in JPDA and PMB filter). Beware the similar data structure used in PMBM can be applied in this case.
    2. Follow the ideas presented at PMBM, handle the missed detection via a two step process instead of leaving things to p_d alone.
"""

from GM_PHD_Filter_Point_Target import GM_PHD_Filter
from util import parse_args, gen_ground_truth_parameters, gen_filter_model, gen_simulation,filter_plot,plot_gospa
from matplotlib import pyplot as plt
import time
import numpy as np
import gospa as gp
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
    Filter = GM_PHD_Filter(filter_model, args.Bayesian_filter_config, args.motion_model_type)
    
    for ith_simulation in range(args.number_of_monte_carlo_simulations):        
        # generate information for this simulation
        Z_k_all, targetStates_all, observations_all, clutter_all = gen_simulation(ground_truth_parameters,args.n_scan,args.simulation_scenario)
        #Initiate data structure for this simulation.
        gospa_record = []
        gospa_localization_record =[]
        gospa_missed_record=[]
        gospa_false_record=[]
        
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
            
            if i == 0: 
                predictedIntensity = Filter.predict_for_initial_step()
            else:
                predictedIntensity = Filter.predict(Z_k, prunedIntensity)
            
            updatedIntensity = Filter.update(Z_k, predictedIntensity)
            prunedIntensity = Filter.pruneAndMerge(updatedIntensity)
            estimates = Filter.extractStates(prunedIntensity)  # Extracting estimates from the pruned intensity this gives better result than extracting them from the updated intensity!
            estimatedStates = estimates['m']
            
            # Store Metrics for Plotting
            gospa,target_to_track_assigments,gospa_localization,gospa_missed,gospa_false = gp.calculate_gospa(targetStates, estimatedStates, c=10.0 , p=2, alpha=2)
            gospa_record.append(gospa)
            if len(target_to_track_assigments)!=0:
                # Normalized locallization error = gospa_localization_error/ number of matched targets 
                gospa_localization_record.append(math.sqrt(gospa_localization)/len(target_to_track_assigments))
            else:
                # TODO  need to check what happen when assignment is 0
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
        print("This is the %dth monte carlo simulation, GM-PHD processing takes %f seconds" %(ith_simulation, (toc - tic)))
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
