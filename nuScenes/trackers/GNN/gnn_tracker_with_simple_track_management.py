import numpy as np
import copy
from trackers.GNN.util import  compute_cost_matrix, Target, TargetMaker
from datetime import datetime
from scipy.optimize import linear_sum_assignment
            
def gnn_tracker(gating_thr,Z_k, filter_model, target_maker, targets, max_id, potential_targets):     

    Z_not_in_gating_regoin_of_any_target_position = [(Z_k[index]['translation'][0], Z_k[index]['translation'][1]) for index in range(len(Z_k))]  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
    Z_not_in_gating_regoin_of_any_target = copy.deepcopy(Z_k)  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
    
    unique_measurements_position = []
    unique_measurements=[]              # The list which will store all the measurements in the gating regions of all target.

    targets_tobe_killed = []                # The list which will store all the targets to be deleted(since for several frames there is no single measurement fall into the gating region of such target).

    for t_idx, t in enumerate(targets):
        # Read = the updated state fo current target from last frame.
        #t_position_before = (t.target_state[0][0], t.target_state[1][0])
        # Perform predict step for current target.
        t.predict()
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
            for m_index, m in enumerate(measurements_inside_gating):
                m_position = (m['translation'][0], m['translation'][1])
                #print(m_position)
                if m_position not in unique_measurements_position:
                    unique_measurements_position.append(m_position)
                    unique_measurements.append(m)
    cost_matrix=[]
    cost_matrix=compute_cost_matrix(targets, unique_measurements)
    row_idx_measurement, col_idx_target=linear_sum_assignment(cost_matrix, maximize=True)

    for u_m in unique_measurements:
        u_m_position = (u_m['translation'][0], u_m['translation'][1])
        if u_m_position in Z_not_in_gating_regoin_of_any_target_position:
            # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
            index_to_be_removed=Z_not_in_gating_regoin_of_any_target_position.index(u_m_position)
            Z_not_in_gating_regoin_of_any_target_position.remove(u_m_position)
            Z_not_in_gating_regoin_of_any_target.remove(Z_not_in_gating_regoin_of_any_target[index_to_be_removed])

    for idx, m_idx in enumerate(row_idx_measurement):
        m=unique_measurements[m_idx]
        m_position = (m['translation'][0], m['translation'][1])
        t_idx=col_idx_target[idx]
        t=targets[t_idx]
        t_position_before = (t.target_state[0][0], t.target_state[1][0])
        #print('measurement at {} is associated with target {} at {}'.format(m_position,targets_id_list[t_idx], t_position_before))
        t.kalman_update(m)
        t_position = (t.target_state[0][0], t.target_state[1][0])
        #print("target {} at {} is updated to {}".format(targets_id_list[t_idx],t_position_before, t_position))
        
    for t_idx, t in enumerate(targets):
        if t_idx not in col_idx_target:
            t.increase_death_counter()  # Increase counter which corresponds to denote "current target should be deleted since it does not exist anymore". 
            death_counter=t.read_death_counter()
            #print('target {} at {} increase death counter to {}'.format(targets_id_list[t_idx], (t.target_state[0][0], t.target_state[1][0]), death_counter))
            if death_counter >= filter_model['death_counter_kill']:
                #print("target {} at {} has been killed".format(targets_id_list[t_idx], (t.target_state[0][0], t.target_state[1][0])))
                targets_tobe_killed.append(t_idx)

    # Track management for tracks: Remove all the death targets.
    if len(targets_tobe_killed)!=0:
        targets_tobe_killed.sort()
        # Delete all the targets in the "targets_tobe_killed" list.
        for offset, dead_target_idx in enumerate(targets_tobe_killed):
            del targets[dead_target_idx-offset]

    # Track management for potential tracks: Generate new potential targets from measurements which have not fall into gating regoin of any target or any potential target.
    if len(Z_not_in_gating_regoin_of_any_target) != 0:
        for z_index, z in enumerate(Z_not_in_gating_regoin_of_any_target):
            max_id+=1
            t=target_maker.new(z, max_id)
            targets.append(t)
            #targets_id_list.append(max_id)
            #print('target {} at {} is born'.format(max_id,(z['translation'][0],z['translation'][1])))
            #print('a potential target is born at {}'.format((z['translation'][0],z['translation'][1])))
    return targets, max_id, potential_targets