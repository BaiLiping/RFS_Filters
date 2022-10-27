"""
Reference: 
    [1] 2009.The Probabilistic Data Association Filter.
        https://www.researchgate.net/publication/224083228_The_probabilistic_data_association_filter
"""
"""
The current version of this code, is updated in 20210831.
ToDo list:
    1) Obviously, the approach used in this demo code to create tracks is NOT proper, and there is no approach used in this demo code to
        manage the disappearing tracks. Thus a proper track management approach is needed for this code. 
        A possible track management approach is to define two levels of track: potential level track and mature level track. We only provide 
        mature tracks as the estimated tracks.
        For all the measurements which have not been associated with any existing tracks, we create the corresponding potential tracks and 
        see if the potential tracks will be associated to measurements and update for N frames. If so, we upgrade the potential track(s) which
        meet the requirement as mentioned above to mature track(s), otherwise delete.
        For all the existing tracks which have not been associated with any measurement, we check if such tracks will be associated to any 
        measurement and update for next N frames. If not, we downgrade such track(s) to potential track(s).
"""

import numpy as np
import cv2 as cv

from cvtargetmaker import CVTargetMaker

# Read the video data. We will use it to provide detection/measurements for tracking multi-targets in image field over frames.
video = cv.VideoCapture("D:/Tech_Resource/Paper_Resource/Signal Processing in General/Random Vector Mutil-Point Target Tracking/JPDA/PDA_PointTarget_Python_Code/PETS09-S2L1.mp4")
bg_subber = cv.createBackgroundSubtractorMOG2()
med_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
small_kernel = np.ones((3, 3), dtype=np.uint8)


Q = 30*np.eye(4, 4)     # Motion noise(process noise) covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
R = 10*np.eye(2, 2)     # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
T = 0.15                # Interval between two frames, T, sets as 0.15 second.
eta = 36                # Gating threshold, eta, which corresponding to the gate probability P_G which is the probability that the gate contains the true measurement if detected. See equation (34) in [1] 
P_D = 0.4               # P_D is the target detection probability, sets as 0.4.
# lambda_clutter is spatial density of clutter(clutter intensity, equal to number_of_clutter_per_frame/FOV_area) under Poisson clutter model
# (Thus in this code we use parametric PDA see equation (38) in [1].).
lambda_clutter = 0.1   # e.g. lambda_clutter = 0.01 denotes "10 clutters per frame in average / 100 square meters".
target_maker = CVTargetMaker(T, Q, R, eta, P_D, lambda_clutter)
have_measurements = False
targets = []
gate = 8

# Run PDA filter over frames.
while True:
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
    cv.imshow('Distance transform', dist)
    cv.threshold(dist, 0.6, 1.0, cv.THRESH_BINARY, dst=dist)
    dist_8u = dist.astype('uint8')
    contours, hierarchy = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    moments = [cv.moments(c) for c in contours]
    Z = [np.array([[np.float(m["m10"] / m["m00"])], [np.float(m["m01"] / m["m00"])]]) for m in moments]
    # Only if the first time we observe more than 4 measurements, we will create targets by directly using thses measurements.
    # This is really BAD idea to initialize tracks, by using such approach the new birth targets later in the video can NOT be handled at all!!!!!
    if len(Z) > 4 and not have_measurements:
        have_measurements = True
        for z in Z:
            targets.append(target_maker.new(z[0][0], z[1][0]))  # Create target by using measurement information directly.
    
    # For each target in the current frame, tracking the multi-targets using PDA filter according to figure 3 in [1].
    for t in targets:
        # Perform Kalman prediction step for current target.
        t.predict()
        eigs = np.linalg.eig(t.P[0:2, 0:2])

        cv.ellipse(img=frame,
                   center=(t.target_state[0], t.target_state[1]),
                   axes=(int(round(np.sqrt(eigs[0][0]))), int(round(np.sqrt(eigs[0][1])))),
                   angle=0,
                   startAngle=0,
                   endAngle=360,
                   color=(255, 0, 0),
                   thickness=2,
                   lineType=1)
        # Perform gating for current target.
        measurements_in_gating_area = t.gating(Z)
        if measurements_in_gating_area.size > 0:
            # For all the measurements inside gating area of current target, perform PDA based data association and update step.
            t.pda_and_update(measurements_in_gating_area)

    # Read all measurements in the currect frame and use circle to denote them(and they will be ploted).
    for z in Z:
        cv.circle(frame, (z[0], z[1]), 3, (0, 255, 0), -1)
    cv.imshow("Frame", frame)

    key = cv.waitKey(30)
    if key == "q" or key == 27:
        break
