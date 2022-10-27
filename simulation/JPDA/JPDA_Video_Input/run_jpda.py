"""
Reference:
    [1] 2009.The Probabilistic Data Association Filter.
        https://www.researchgate.net/publication/224083228_The_probabilistic_data_association_filter
    [2] 1983.Sonar Tracking of Multiple Targets Using Joint Probabilistic Data Association.

The current version of this code, is updated in 20210830.
"""
# @TODO: We should correct the DPA update, by following equation (37), (38) in [1] exactly.

import numpy as np
import cv2 as cv
from numpy.lib.arraysetops import unique
from numpy.testing._private.utils import measure

from cvtargetmaker import CVTargetMaker
from utils import generate_validation_matrix, generate_feasible_assiciation_matrices_of_joint_events, construct_probability_dictionary, compute_joint_probabilities, compute_marginal_probability

# Read the video data. We will use it to provide measurements/detections for tracking multi-targets in image field over frames.
video = cv.VideoCapture("D:/Tech_Resource/Paper_Resource/Signal Processing in General/Random Vector Mutil-Point Target Tracking/JPDA/JPDA_PointTarget_Python_Code/PETS09-S2L1.mp4")
bg_subber = cv.createBackgroundSubtractorMOG2()
med_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
small_kernel = np.ones((3, 3), dtype=np.uint8)

# Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
Q = 30*np.eye(4, 4)
# Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
R = 10*np.eye(2, 2)
T = 0.2                # Interval between two frames, T, sets as 0.3 second.
# Gating threshold, eta, which corresponding to the gate probability P_G which is the probability that the gate contains the true measurement if detected. See equation (34) in [1]
eta = 4                 # Gating threshold.
P_D = 0.95               # P_D is the target detection probability.
# lambda_clutter is spatial density of clutter(clutter intensity, equal to number_of_clutter_per_frame/FOV_area) under Poisson clutter model
# (Thus in this code we use parametric JPDA, see equation (47) in [1], and parametric PDA see equation (38) in [1].).
lambda_clutter = 0.1   # e.g. lambda_clutter = 0.01 denotes "10 clutters per frame in average / 100 square meters".
# The target maker which returns the CVTarget, which runs the core part of Kalman + JPDA.
target_maker = CVTargetMaker(T, Q, R, eta, P_D, lambda_clutter)

targets = []
potential_death = []
potential_targets = []
death_counter_kill = 5  # The threshold of counter to trigger the deletion of target.
birth_counter_born = 5  # The threshold of counter to trigger "potential target becomes track".
birth_initiation = 3

counting_list = []
measurement_outside_of_gating = []
measurement_outside_of_gating_set = []

np.random.seed(10)  # Set seed to make the implementation reproducable.

# Run JPDA filter over frames.
while True:
    # Generate Input Data for current frame.
    ret, frame = video.read()
    if frame is None:
        break
    fg = bg_subber.apply(frame)
    cv.threshold(fg, 200, 255, cv.THRESH_BINARY, dst=fg)
    cleaned = cv.morphologyEx(fg, cv.MORPH_OPEN, med_kernel)
    dist, labels = cv.distanceTransformWithLabels(cleaned, cv.DIST_L2, 2)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    dist = cv.dilate(dist, med_kernel)
    dist = cv.morphologyEx(dist, cv.MORPH_DILATE, med_kernel, iterations=2)
    #cv.imshow('Distance transform', dist)
    cv.threshold(dist, 0.6, 1.0, cv.THRESH_BINARY, dst=dist)
    dist_8u = dist.astype('uint8')
    contours, hierarchy = cv.findContours(
        dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    moments = [cv.moments(c) for c in contours]

    # Z is the input data. Notice the format is array[array[],array[]].
    Z = [np.array([[np.float(m["m10"] / m["m00"])],
                  [np.float(m["m01"] / m["m00"])]]) for m in moments]

    Z_not_in_gating_regoin_of_any_target = [(z[0][0], z[1][0]) for z in Z]  # Initiate a copy of Z, firstly just put all the measurements into category "not in gating region of any target".
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
        measurements_inside_gating = t.gating(Z)
        # Compute the likelihood of each measurements within gating region of current target by using Gaussian 
        # pdf likelihood function L_i(k)(as how it is calculated in equation (38) or in equation (48) in [1].). 
        # I don't understand why we need this here.
        t.compute_likelihood_probability_for_each_measurement(measurements_inside_gating)
        # Read the predicted state for current target at this frame.
        t_position = (t.target_state[0][0], t.target_state[1][0])
        print("target at {} predicted to {}, the measurements fall into gating region of this target is: ".format(
            t_position_before, t_position))
        if len(measurements_inside_gating) != 0:           
            if t.read_death_counter() != 0:  # Check if this target is on the potential death list.
                t.decrease_death_counter()  # If so, decrease its counter by one.
            for m in measurements_inside_gating:
                m_position = (m[0][0], m[1][0])
                print(m_position)
                if m_position in Z_not_in_gating_regoin_of_any_target:
                    # Remove each measurement in the gating region of current target from the category "not in gating region of any target".
                    Z_not_in_gating_regoin_of_any_target.remove(m_position)
                if m_position not in unique_measurements:
                    unique_measurements.append(m_position)
        else:  # When there is no measurements fall into gating region of current target.
            t.increase_death_counter()  # Increase counter which corresponds to denote "current target should be deleted since it does not exist anymore". 
            print('there is no measurements fall into gating region of this target at {}.'.format(t_position))
            if t.read_death_counter() == death_counter_kill:
                # Leave this target into "targets_tobe_killed" list if the counter meets threshold.
                print("Track at {} has been killed".format(
                    (t.target_state[0][0], t.target_state[1][0])))
                targets_tobe_killed.append(t)

    # Track management for tracks: Remove all the death targets.
    if len(targets_tobe_killed)!=0:
        # Delete all the targets in the "targets_tobe_killed" list.
        for dead_target in targets_tobe_killed:
            targets.remove(dead_target)            

    """
    Step 2: Generate marginal probability(Equation 3.19 and 3.20 of [2]) of every measurement in the gating region of each target, for JPDA data association based update.
    """ 
    # Note that we could have just used a single function to implement all these functionality as below, but here we divide them into different functions for the purpose of understanding JPDA data association better.
    validation_matrix = generate_validation_matrix(targets, unique_measurements)  # Generate a validation matrix according to figure 1 and equation (3.5) of [2].
    if len(validation_matrix) != 0 and len(validation_matrix[0]) != 0:
        feasible_association_matrices_of_joint_events = generate_feasible_assiciation_matrices_of_joint_events(validation_matrix) # Generate association matrix for every feasible joint event according to the instructions detailed in page 4 of [2].
        probability_dictionary = construct_probability_dictionary(targets, unique_measurements)   # Reconstruct the calculated Gaussian pdf likelihood of every measurement in the regoin area of every target into a dictionary.
        joint_probabilities = compute_joint_probabilities(targets, feasible_association_matrices_of_joint_events, probability_dictionary, P_D, lambda_clutter) # Compute joint probability according to equation 3.18 of [2].
        marginal_probability = compute_marginal_probability(targets, feasible_association_matrices_of_joint_events, joint_probabilities, unique_measurements) # Compute marginal probability according to equation 3.19 and 3.20 of [2].

    """
    Step 3: Perform JPDA based data association and update step for every existing target.
    """
    for t in targets:
        t_position_before = (t.target_state[0][0], t.target_state[1][0])
        measurements_in_gating_area = t.read_measurements_within_gating() # generate all the measurements associated with current target
        if len(measurements_in_gating_area) >0:
            # For all the measurements inside gating area of current target, perform JPDA based data association and update step.
            t.jpda_update(marginal_probability, measurements_in_gating_area)
            t_position = (t.target_state[0][0], t.target_state[1][0])
            print("target at {} updated to {}".format(t_position_before, t_position))
        # Illustrate every updated track.
        eigs = np.linalg.eig(t.P[0:2, 0:2])
        cv.ellipse(img=frame,
                    center=(int(t.target_state[0][0]), int(t.target_state[1][0])),
                    axes=(int(round(np.sqrt(eigs[0][0]))), int(round(np.sqrt(eigs[0][1])))),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 255), # red ellipse for tracking mature tracks.
                    thickness=2,
                    lineType=1)

    """
    Step 4: Track management for all the potential tracks.
    """
    # Track management for potential tracks: Perform predict step and check which measurements fall into the gating region for every existing potential target.
    Z_potential = [z for z in Z if (z[0][0], z[1][0]) in Z_not_in_gating_regoin_of_any_target]
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
            print('measurements fall into gating region of potential target at {} is: '.format(p_t_position))
            p_t.increase_birth_counter()  # If so, decrease its counter by one
            if p_t.read_birth_counter() == birth_counter_born:
                targets.append(p_t)
                potential_targets_tobe_killed.append(p_t)
                #potential_targets.remove(p_t)
                for m in measurements_inside_gating:
                    m_position = (m[0][0], m[1][0])
                    print(m_position)
                    if m_position not in unique_measurements:
                        unique_measurements.append(m_position)
                print("Potential Track at {} has become a track.".format((p_t.target_state[0][0], p_t.target_state[1][0])))

            for m in measurements_inside_gating:
                position = (m[0][0], m[1][0])                  
                if position in Z_not_in_gating_regoin_of_any_target:
                    # Remove each measurement in the gating region of current potential target from the category "not in gating region of any target".
                    Z_not_in_gating_regoin_of_any_target.remove(position)

        else:  # When there is no measurement fall into gating region of this potential track.
            p_t.decrease_birth_counter()    # Decrease the counter which corresponds to denote "current potential target should be converted to track". 
            if p_t.read_birth_counter() == 0:
                # Leave this potential target into "potential_targets_tobe_killed" list if the counter == 0.
                print("Potential Track at {} has been killed".format(
                    (p_t.target_state[0][0], p_t.target_state[1][0])))
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
            print("A Potential Track at {} has been initiated".format((z[0],z[1])))
        
    # Print out information for targets and potential tracks in current frame.
    print('-'*50)
    print('this frame has {} measurements'.format(len(Z)))
    print('there are {} targets to be tracked'.format(len(targets)))
    print('there are {} potential targets to be tracked'.format(len(potential_targets)))
    print('-'*50)

    # Track management for potential tracks: Perform PDA based data association and update step for every existing potential target.
    for p_t in potential_targets:
        eigs = np.linalg.eig(p_t.P[0:2, 0:2])
        measurements_in_gating_area = p_t.read_measurements_within_gating()
        if len(measurements_in_gating_area) > 0:
            p_t.pda_update(measurements_in_gating_area)
        # Illustrate every updated potential track.
        cv.ellipse(img=frame,
                   center=(int(p_t.target_state[0][0]), int(p_t.target_state[1][0])),
                   axes=(int(round(np.sqrt(eigs[0][0]))), int(round(np.sqrt(eigs[0][1])))),
                   angle=0,
                   startAngle=0,
                   endAngle=360,
                   color=(255, 0, 0),  # blue ellipse for tracking potential targets
                   thickness=1,
                   lineType=1)

    """
    Step 5: Plot and other specific parts for opencv stuff.
    """
    # Read all measurements in the currect frame and use circle to denote them(and they will be ploted).
    for z in Z:
        cv.circle(img = frame, 
                    center = (int(z[0][0]), int(z[1][0])), 
                    radius = 3, 
                    color = (0, 255, 0),
                    thickness = -1)

    cv.imshow("Frame", frame)

    key = cv.waitKey(30)
    if key == "q" or key == 27:
        break
