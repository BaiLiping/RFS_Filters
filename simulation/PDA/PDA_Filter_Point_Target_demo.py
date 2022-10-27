"""
Reference:
    [1] 2009.The Probabilistic Data Association Filter.
        https://www.researchgate.net/publication/224083228_The_probabilistic_data_association_filter
    [2] 1983.Sonar Tracking of Multiple Targets Using Joint Probabilistic Data Association.

The current version of this code, is updated in 20210830.
"""

import numpy as np
from numpy.lib.arraysetops import unique
from numpy.testing._private.utils import measure
from util import Target, TargetMaker, parse_args, gen_ground_truth_parameters, gen_filter_model, gen_simulation,filter_plot,plot_gospa
from matplotlib import pyplot as plt
import time
import numpy as np
import gospa as gp
import math

targets = []
potential_death = []
potential_targets = []


counting_list = []
measurement_outside_of_gating = []
measurement_outside_of_gating_set = []

def main(args):
    # Initiate data structure for GOSPA record
    gospa_record_all = []
    gospa_localization_record_all =[]
    gospa_missed_record_all=[]
    gospa_false_record_all=[]

    # Generate simulation
    ground_truth_parameters = gen_ground_truth_parameters()

    # Generate parameters for filter model
    filter_model = gen_filter_model()
    target_maker = TargetMaker(filter_model['T'], filter_model['Q'], filter_model['R'], filter_model['eta'], filter_model['P_D'], filter_model['clutterIntensity'], args.birth_initiation)

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

            Z_not_in_gating_regoin_of_any_target = [(z[0][0], z[1][0]) for z in Z_k]  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
            unique_measurements = []                # The list which will store all the measurements in the gating regions of all target.
            targets_tobe_killed = []                # The list which will store all the targets to be deleted(since for several frames there is no single measurement fall into the gating region of such target).
            potential_targets_tobe_killed = []      # The list which will store all the potential targets to be deleted(since for several frames there is no single measurement fall into the gating region of such potential target).
        
            """
            Step 1: Perform predict step and check which measurements fall into the gating region for every existing target.
            """
            for t in targets:
                # Read = the updated state fo current target from last frame.
                t_position_before = (t.target_state[0][0], t.target_state[1][0])
                # Perform predict step for current target.
                t.predict()
                # Clear the registry for measurements association corresponds to current target.
                t.clear_measurements_association()
                # Decide which meansurements inside the gating region of current target.
                measurements_inside_gating = t.gating(Z_k)
                # Compute the likelihood of each measurements within gating region of current target by using Gaussian 
                # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
                t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
                # Read the predicted state for current target at this frame.
                t_position = (t.target_state[0][0], t.target_state[1][0])
                #print("target at {} predicted to {}, the measurements fall into gating region of this target is: ".format(t_position_before, t_position))
                if len(measurements_inside_gating) != 0:           
                    if t.read_death_counter() != 0:  # Check if this target is on the potential death list.
                        t.decrease_death_counter()  # If so, decrease its counter by one.
                    for m in measurements_inside_gating:
                        m_position = (m[0][0], m[1][0])
                        #print(m_position)
                        if m_position in Z_not_in_gating_regoin_of_any_target:
                            # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
                            Z_not_in_gating_regoin_of_any_target.remove(m_position)
                        if m_position not in unique_measurements:
                            unique_measurements.append(m_position)
                else:  # When there is no measurements fall into gating region of current target.
                    t.increase_death_counter()  # Increase counter which corresponds to denote "current target should be deleted since it does not exist anymore". 
                    #print('there is no measurements fall into gating region of this target at {}.'.format(t_position))
                    if t.read_death_counter() == args.death_counter_kill:
                        # Leave this target into "targets_tobe_killed" list if the counter meets threshold.
                        #print("Track at {} has been killed".format((t.target_state[0][0], t.target_state[1][0])))
                        targets_tobe_killed.append(t)
        
            # Track management for tracks: Remove all the death targets.
            if len(targets_tobe_killed)!=0:
                # Delete all the targets in the "targets_tobe_killed" list.
                for dead_target in targets_tobe_killed:
                    targets.remove(dead_target)            
    
            """
            Step 2: Perform PDA based data association and update step for every existing target.
            """
            estimatedStates = []
            for t in targets:
                measurements_in_gating_area = t.read_measurements_within_gating() # generate all the measurements associated with current target
                if len(measurements_in_gating_area) >0:
                    # For all the measurements inside gating area of current target, perform PDA based data association and update step.
                    t.pda_update(measurements_in_gating_area)
                    t_position = (t.target_state[0][0], t.target_state[1][0])
                    estimatedStates.append(t_position)
                    #print("target at {} updated to {}".format(t_position_before, t_position))

            """
            Step 3: Track management for all the potential tracks.
            """
            # Track management for potential tracks: Perform predict step and check which measurements fall into the gating region for every existing potential target.
            Z_potential = [z for z in Z_k if (z[0][0], z[1][0]) in Z_not_in_gating_regoin_of_any_target]
            for p_t in potential_targets:
                # Perform predict step for current potential target.
                p_t.predict()
                # Clear the registry for measurements association corresponds to current potential target.
                p_t.clear_measurements_association()
                # Decide which meansurements inside the gating region of current potential target.
                measurements_inside_gating = p_t.gating(Z_potential)
                # Compute the likelihood of each measurements within gating region of current potential target by using Gaussian 
                # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
                p_t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
                # Read the predicted state for current potential target at this frame.
                p_t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
                if len(measurements_inside_gating) != 0:
                    #print('measurements fall into gating region of potential target at {} is: '.format(p_t_position))
                    p_t.increase_birth_counter()  # If so, decrease its counter by one
                    if p_t.read_birth_counter() == args.birth_counter_born:
                        targets.append(p_t)
                        potential_targets_tobe_killed.append(p_t)
                        #potential_targets.remove(p_t)
                        for m in measurements_inside_gating:
                            m_position = (m[0][0], m[1][0])
                            #print(m_position)
                            if m_position not in unique_measurements:
                                unique_measurements.append(m_position)
                        t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
                        #print("Potential Track at {} has become a track.".format(t_position))
                        # for mature track, add it to estimated state
                        estimatedStates.append(t_position)

                    for m in measurements_inside_gating:
                        position = (m[0][0], m[1][0])                  
                        if position in Z_not_in_gating_regoin_of_any_target:
                            # Remove each measurement in the gating region of current potential target from the category "not in gating region of any target".
                            Z_not_in_gating_regoin_of_any_target.remove(position)
        
                else:  # When there is no measurement fall into gating region of this potential track.
                    p_t.decrease_birth_counter()    # Decrease the counter which corresponds to denote "current potential target should be converted to track". 
                    if p_t.read_birth_counter() == 0:
                        # Leave this potential target into "potential_targets_tobe_killed" list if the counter == 0.
                        #print("Potential Track at {} has been killed".format((p_t.target_state[0][0], p_t.target_state[1][0])))
                        #potential_targets.remove(p_t)
                        potential_targets_tobe_killed.append(p_t)
            
            # Track management for potential tracks: Remove all the death potential targets.
            if len(potential_targets_tobe_killed) != 0:
                for dead_p_t in potential_targets_tobe_killed:
                    potential_targets.remove(dead_p_t)
                
            # Track management for potential tracks: Generate new potential targets from measurements which have not fall into gating regoin of any target or any potential target.
            if len(Z_not_in_gating_regoin_of_any_target) != 0:
                for z in Z_not_in_gating_regoin_of_any_target:
                    potential_targets.append(target_maker.new(z[0], z[1]))
                    t_position = (z[0],z[1])
                    #print("A Potential Track at {} has been initiated".format(t_position))
                
            # Print out information for targets and potential tracks in current frame.
            #print('-'*50)
            #print('this frame has {} measurements'.format(len(Z_k)))
            #print('there are {} targets to be tracked'.format(len(targets)))
            #print('there are {} potential targets to be tracked'.format(len(potential_targets)))
            #print('-'*50)
                    
            # Store Metrics for Plotting
            gospa,target_to_track_assigments,gospa_localization,gospa_missed,gospa_false = gp.calculate_gospa(targetStates, estimatedStates, c=10.0 , p=2, alpha=2)
            gospa_record.append(gospa)
            if len(target_to_track_assigments)!=0:
                # Normalized locallization error = gospa_localization_error/ number of matched targets 
                gospa_localization_record.append(math.sqrt(gospa_localization)/len(target_to_track_assigments))
            else:
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
        print("This is the %dth monte carlo simulation, PDA processing takes %f seconds" %(ith_simulation, (toc - tic)))
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
