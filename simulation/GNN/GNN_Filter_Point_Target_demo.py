import numpy as np
from numpy.lib.arraysetops import unique
from numpy.testing._private.utils import measure
from util import Target, TargetMaker, parse_args, gen_ground_truth_parameters, gen_filter_model, gen_simulation,filter_plot,plot_gospa, generate_validation_matrix, generate_feasible_assiciation_matrices_of_joint_events, construct_probability_dictionary, compute_joint_probabilities, compute_marginal_probability
from matplotlib import pyplot as plt
import time
import numpy as np
import gospa as gp
import math
import copy

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

        Z_not_in_gating_regoin_of_any_target_position = [(Z_k[index]['translation'][0], Z_k[index]['translation'][1]) for index in range(len(Z_k))]  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
        Z_not_in_gating_regoin_of_any_target = copy.deepcopy(Z_k)  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
        
        unique_measurements_position = []
        unique_measurements=[]              # The list which will store all the measurements in the gating regions of all target.
    
        targets_tobe_killed = []                # The list which will store all the targets to be deleted(since for several frames there is no single measurement fall into the gating region of such target).
        potential_targets_tobe_killed = []      # The list which will store all the potential targets to be deleted(since for several frames there is no single measurement fall into the gating region of such potential target).
    
        for t_idx, t in enumerate(targets):
            t.predict()
            # Clear the registry for measurements association corresponds to current target.
            t.clear_measurements_association()
            # Decide which meansurements inside the gating region of current target.
            measurements_inside_gating = t.gating(gating_thr,Z_k)
            # Compute the likelihood of each measurements within gating region of current target by using Gaussian 
            # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
            t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
            # Read the predicted state for current target at this frame.
            if len(measurements_inside_gating) != 0:           
                for m in measurements_inside_gating:
                    m_position = (m['translation'][0], m['translation'][1])
                    if m_position not in unique_measurements_position:
                        unique_measurements_position.append(m_position)
                        unique_measurements.append(m)
        
        for u_m in unique_measurements:
            u_m_position = (u_m['translation'][0], u_m['translation'][1])
            if u_m_position in Z_not_in_gating_regoin_of_any_target_position:
                # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
                index_to_be_removed=Z_not_in_gating_regoin_of_any_target_position.index(u_m_position)
                Z_not_in_gating_regoin_of_any_target_position.remove(u_m_position)
                Z_not_in_gating_regoin_of_any_target.remove(Z_not_in_gating_regoin_of_any_target[index_to_be_removed])
    
        cost_matrix=[]
        cost_matrix=compute_cost_matrix(targets, unique_measurements)
        row_idx_measurement, col_idx_target=linear_sum_assignment(cost_matrix, maximize=True)
    
    
        for idx, m_idx in enumerate(row_idx_measurement):
            m=unique_measurements[m_idx]
            m_position = (m['translation'][0], m['translation'][1])
            t_idx=col_idx_target[idx]
            t=targets[t_idx]
            t.kalman_update(m)
        for t_idx, t in enumerate(targets):
            if t_idx not in col_idx_target:           
                t.increase_death_counter()  # Increase counter which corresponds to denote "current target should be deleted since it does not exist anymore". 
                if t.read_death_counter() >= filter_model['death_counter_kill']:
                    targets_tobe_killed.append(t_idx)
    
        # Track management for tracks: Remove all the death targets.
        if len(targets_tobe_killed)!=0:
            targets_tobe_killed.sort()
            # Delete all the targets in the "targets_tobe_killed" list.
            for offset, dead_target_idx in enumerate(targets_tobe_killed):
                potential_targets.append(targets[dead_target_idx-offset])
                del targets[dead_target_idx-offset]
    
    
        # Track management for potential tracks: Perform predict step and check which measurements fall into the gating region for every existing potential target.
        unique_measurements_position_potential=[]
        unique_measurements_potential=[]
        potential_targets_tobe_killed = [] 
    
        for p_t_idx, p_t in enumerate(potential_targets):
            # Perform predict step for current potential target.
            p_t.predict()
            # Clear the registry for measurements association corresponds to current potential target.
            p_t.clear_measurements_association()
            # Decide which meansurements inside the gating region of current potential target.
            measurements_inside_gating = p_t.gating(gating_thr,Z_not_in_gating_regoin_of_any_target)
            # Compute the likelihood of each measurements within gating region of current potential target by using Gaussian 
            # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
            p_t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
            # Read the predicted state for current potential target at this frame.
            p_t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
            if len(measurements_inside_gating) != 0:
                for m in measurements_inside_gating:
                    m_position = (m['translation'][0], m['translation'][1])
                    if m_position not in unique_measurements_position_potential:
                        unique_measurements_position_potential.append(m_position)
                        unique_measurements_potential.append(m)
        cost_matrix=[]
        cost_matrix=compute_cost_matrix(potential_targets, unique_measurements_potential)
        row_idx_measurement, col_idx_target=linear_sum_assignment(cost_matrix, maximize=True)
    
        for u_m in unique_measurements_potential:
            u_m_position = (u_m['translation'][0], u_m['translation'][1])
            if u_m_position in Z_not_in_gating_regoin_of_any_target_position:
                # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
                index_to_be_removed=Z_not_in_gating_regoin_of_any_target_position.index(u_m_position)
                Z_not_in_gating_regoin_of_any_target_position.remove(u_m_position)
                Z_not_in_gating_regoin_of_any_target.remove(Z_not_in_gating_regoin_of_any_target[index_to_be_removed])
    
        for idx, m_idx in enumerate(row_idx_measurement):
            m=unique_measurements_potential[m_idx]
            m_position = (m['translation'][0], m['translation'][1])
            p_t_idx=col_idx_target[idx]
            p_t=potential_targets[p_t_idx]
            p_t_position_before = (p_t.target_state[0][0], p_t.target_state[1][0])
            p_t.kalman_update(m)
            p_t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
            targets.append(p_t)
            potential_targets_tobe_killed.append(p_t_idx)
    
        # Track management for potential tracks: Remove all the death potential targets.
        if len(potential_targets_tobe_killed) != 0:
            potential_targets_tobe_killed.sort()
            for offset, dead_p_t_idx in enumerate(potential_targets_tobe_killed):
                del potential_targets[dead_p_t_idx-offset]
    
        # Track management for potential tracks: Generate new potential targets from measurements which have not fall into gating regoin of any target or any potential target.
        if len(Z_not_in_gating_regoin_of_any_target) != 0:
            for z_index, z in enumerate(Z_not_in_gating_regoin_of_any_target):
                max_id+=1
                targets.append(target_maker.new(z, max_id))   
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
        print("This is the %dth monte carlo simulation, JPDA processing takes %f seconds" %(ith_simulation, (toc - tic)))
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
