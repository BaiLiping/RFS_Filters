import numpy as np
import copy
from trackers.JPDA.util import  compute_cost_matrix, compute_marginals, Target, TargetMaker, gen_filter_model, generate_validation_matrix, generate_feasible_assiciation_matrices_of_joint_events, construct_probability_dictionary, compute_joint_probabilities, compute_marginal_probability
from datetime import datetime
from scipy.optimize import linear_sum_assignment
            
def run_jpda_tracker(gating_thr,Z_k, filter_model, target_maker, targets, targets_id_list, max_id, potential_targets):     

    Z_not_in_gating_regoin_of_any_target_position = [(Z_k[index]['translation'][0], Z_k[index]['translation'][1]) for index in range(len(Z_k))]  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
    Z_not_in_gating_regoin_of_any_target = copy.deepcopy(Z_k)  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
    
    unique_measurements_position = []
    unique_measurements=[]              # The list which will store all the measurements in the gating regions of all target.

    targets_tobe_killed = []                # The list which will store all the targets to be deleted(since for several frames there is no single measurement fall into the gating region of such target).
    potential_targets_tobe_killed = []      # The list which will store all the potential targets to be deleted(since for several frames there is no single measurement fall into the gating region of such potential target).

    """
    Step 1: Perform predict step and check which measurements fall into the gating region for every existing target.
    """
    for t_idx, t in enumerate(targets):
        # Read = the updated state fo current target from last frame.
        #t_position_before = (t.target_state[0][0], t.target_state[1][0])
        # Perform predict step for current target.
        t.predict()
        target_position = (t.target_state[0][0], t.target_state[1][0])
        # Clear the registry for measurements association corresponds to current target.
        t.clear_measurements_association()
        # Decide which meansurements inside the gating region of current target.
        measurements_inside_gating = t.gating(gating_thr,Z_k)
        # Compute the likelihood of each measurements within gating region of current target by using Gaussian 
        # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
        t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
        # Read the predicted state for current target at this frame.
        #print("target at {} predicted to {}, the measurements fall into gating region of this target is: ".format(t_position_before, t_position))
        if len(measurements_inside_gating) != 0:           
            #if t.read_death_counter() != 0:  # Check if this target is on the potential death list.
            #    t.decrease_death_counter()  # If so, decrease its counter by one.
            for m_index, m in enumerate(measurements_inside_gating):
                m_position = (m['translation'][0], m['translation'][1])
                print('{} is associated with target {} at {}'.format(m_position,targets_id_list[t_idx], target_position))
                #print(m_position)
                if m_position in Z_not_in_gating_regoin_of_any_target_position:
                    position_index=Z_not_in_gating_regoin_of_any_target_position.index(m_position)
                    # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
                    index_to_be_removed=Z_not_in_gating_regoin_of_any_target_position.index(m_position)
                    Z_not_in_gating_regoin_of_any_target_position.remove(m_position)
                    Z_not_in_gating_regoin_of_any_target.remove(Z_not_in_gating_regoin_of_any_target[index_to_be_removed])
                if m_position not in unique_measurements_position:
                    unique_measurements_position.append(m_position)
                    unique_measurements.append(m_index)
        else:  # When there is no measurements fall into gating region of current target.
            t.increase_death_counter()  # Increase counter which corresponds to denote "current target should be deleted since it does not exist anymore". 
            #print('there is no measurements fall into gating region of this target at {}.'.format(t_position))
            if t.read_death_counter() >= filter_model['death_counter_kill']:
                # Leave this target into "targets_tobe_killed" list if the counter meets threshold.
                print("target {} at {} has been killed".format(targets_id_list[t_idx], (t.target_state[0][0], t.target_state[1][0])))
                targets_tobe_killed.append(t_idx)

    # Track management for tracks: Remove all the death targets.
    if len(targets_tobe_killed)!=0:
        # Delete all the targets in the "targets_tobe_killed" list.
        for offset, dead_target_idx in enumerate(targets_tobe_killed):
            #dead_target_position=(dead_target.target_state[0][0], dead_target.target_state[1][0])
            #for t_index, t in enumerate(targets):
            #    t_position = (t.target_state[0][0], t.target_state[1][0])
            #    if dead_target_position==t_position:
            #        del targets[t_index]
            #        del targets_id_list[t_index]  
            del targets[dead_target_idx-offset]
            del targets_id_list[dead_target_idx-offset]         

    """
    Step 2: Generate marginal probability(Equation 3.19 and 3.20 of [2]) of every measurement in the gating region of each target, for JPDA data association based update.
    """ 
    ## Note that we could have just used a single function to implement all these functionality as below, but here we divide them into different functions for the purpose of understanding JPDA data association better.
    #validation_matrix = generate_validation_matrix(targets, unique_measurements_position)  # Generate a validation matrix according to figure 1 and equation (3.5) of [2].
    #if len(validation_matrix) != 0 and len(validation_matrix[0]) != 0:
    #    feasible_association_matrices_of_joint_events = generate_feasible_assiciation_matrices_of_joint_events(validation_matrix) # Generate association matrix for every feasible joint event according to the instructions detailed in page 4 of [2].
    #    probability_dictionary = construct_probability_dictionary(targets, unique_measurements_position)   # Reconstruct the calculated Gaussian pdf likelihood of every measurement in the regoin area of every target into a dictionary.
    #    joint_probabilities = compute_joint_probabilities(targets, feasible_association_matrices_of_joint_events, probability_dictionary, filter_model['P_D'], filter_model['clutterIntensity']) # Compute joint probability according to equation 3.18 of [2].
    #    marginal_probability = compute_marginal_probability(targets, feasible_association_matrices_of_joint_events, joint_probabilities, unique_measurements_position) # Compute marginal probability according to equation 3.19 and 3.20 of [2].

    marginal_probability=compute_marginals(targets)
    """
    Step 3: Perform JPDA based data association and update step for every existing target.
    """
    for t_idx, t in enumerate(targets):
        t_position_before = (t.target_state[0][0], t.target_state[1][0])
        measurements_in_gating_area = t.read_measurements_within_gating() # generate all the measurements associated with current target
        if len(measurements_in_gating_area) >0:
            # For all the measurements inside gating area of current target, perform JPDA based data association and update step.
            t.jpda_update(marginal_probability, measurements_in_gating_area)
            t_position = (t.target_state[0][0], t.target_state[1][0])
            #estimatedStates.append(t_position)
            print("target {} at {} updated to {}".format(targets_id_list[t_idx], t_position_before,t_position))
    """
    Step 4: Track management for all the potential tracks.
    """
    # Track management for potential tracks: Perform predict step and check which measurements fall into the gating region for every existing potential target.
    Z_potential=[]
    Z_potential_position = []
    for z_index, z in enumerate(Z_k): 
        if (z['translation'][0], z['translation'][1]) in Z_not_in_gating_regoin_of_any_target_position:
            Z_potential_position.append((z['translation'][0], z['translation'][1]))
            Z_potential.append(z)

    for p_t_idx, p_t in enumerate(potential_targets):
        # Perform predict step for current potential target.
        p_t.predict()
        # Clear the registry for measurements association corresponds to current potential target.
        p_t.clear_measurements_association()
        # Decide which meansurements inside the gating region of current potential target.
        measurements_inside_gating = p_t.gating(gating_thr,Z_potential)
        # Compute the likelihood of each measurements within gating region of current potential target by using Gaussian 
        # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
        p_t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
        # Read the predicted state for current potential target at this frame.
        p_t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
        if len(measurements_inside_gating) != 0:
            p_t.pda_update(measurements_inside_gating)
            p_t_position_update = (p_t.target_state[0][0], p_t.target_state[1][0])
            print("potential target at {} updated to {}".format(p_t_position, p_t_position_update))
            #print('measurements fall into gating region of potential target at {} is: '.format(p_t_position))
            p_t.increase_birth_counter()  # If so, decrease its counter by one
            if p_t.read_birth_counter() >= filter_model['birth_counter_born']:
                targets.append(p_t)
                max_id+=1
                targets_id_list.append(max_id)
                print('target {} is born'.format(max_id))
                potential_targets_tobe_killed.append(p_t_idx)
                #potential_targets.remove(p_t)
                for m_index, m in enumerate(measurements_inside_gating):
                    m_position = (m['translation'][0], m['translation'][1])
                    #print(m_position)
                    if m_position not in unique_measurements:
                        unique_measurements_position.append(m_position)
                        unique_measurements.append(m)

            for m_index, m in enumerate(measurements_inside_gating):
                m_position = (m['translation'][0], m['translation'][1])
                print('measurement {} is associated with potential target{}'.format(m_position, p_t_position))
                if m_position in Z_not_in_gating_regoin_of_any_target_position:
                    position_index=Z_not_in_gating_regoin_of_any_target_position.index(m_position)
                    # Remove each measurement in the gating region of current potential target from the category "not in gating region of any target".
                    
                    Z_not_in_gating_regoin_of_any_target.remove(Z_not_in_gating_regoin_of_any_target[position_index])
                    Z_not_in_gating_regoin_of_any_target_position.remove(m_position)
                                   

        else:  # When there is no measurement fall into gating region of this potential track.
            p_t.decrease_birth_counter()    # Decrease the counter which corresponds to denote "current potential target should be converted to track". 
            if p_t.read_birth_counter() <= 0:
                # Leave this potential target into "potential_targets_tobe_killed" list if the counter == 0.
                print("Potential Track at {} has been killed".format((p_t.target_state[0][0], p_t.target_state[1][0])))
                #potential_targets.remove(p_t)
                potential_targets_tobe_killed.append(p_t_idx)
    
    # Track management for potential tracks: Remove all the death potential targets.
    if len(potential_targets_tobe_killed) != 0:
        for offset, dead_p_t_idx in enumerate(potential_targets_tobe_killed):
            #dead_p_t_position=(dead_p_t.target_state[0][0], dead_p_t.target_state[1][0])
            #for p_t_index, p_t in enumerate(potential_targets):
            #    p_t_position = (p_t.target_state[0][0], p_t.target_state[1][0])
            #    if dead_p_t_position==p_t_position:
            #        del potential_targets[p_t_index]
            del potential_targets[dead_p_t_idx-offset]

    # Track management for potential tracks: Generate new potential targets from measurements which have not fall into gating regoin of any target or any potential target.
    if len(Z_not_in_gating_regoin_of_any_target) != 0:
        for z_index, z in enumerate(Z_not_in_gating_regoin_of_any_target):

            potential_targets.append(target_maker.new(z))
            print('a potential target is born at {}'.format((z['translation'][0],z['translation'][1])))
    return targets, targets_id_list, max_id, potential_targets