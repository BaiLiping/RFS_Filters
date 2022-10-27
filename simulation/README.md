# This is the Python code for simulation of various point target based MOT trackers including DPA, JDPA, PHD, CPHD, PMBM.

## Parameters for MOT filters:
1. **p_s**: probability of survival. (1-p_s) describe the probability of track death.
2. **birth_intensity**: how likely it is for a track to be born. 
2. **p_d**: probability of detection. (1-p_d) describe the probability of missed detection.
3. **clutter_intensity**: how likely it is for the environment to generate clutter measurement. For instance, it is more likely to have clutters in a closed room than does open spaces.

## Problems the MOT filters try to solve:
1. **missed detection** even if there are no measurement associted with this track or the potential track, doesn't mean the track dose not exist. (1-p_d)w gives leeway such that even this track has no associations for a few frames, it would remain a valid track.
2. **deteced clutter** even when there are measurements associated with this track, does not mean that this is a true measurement. It could very well be clutters. It is solved by scaling things with clutter_intensity. A potential track should not be elevated to a true track with just one measurement association. 
4. **sudden death, sudden birth** this is recolved with p_s,birth_intensity. 
3. **data association** this would be either solved by global permutation optimization (GNN); weighted sum (PDA); weighted sum without double association (JPDA) or the Random Finite Set based methods. 

## Statistical Filter
1. **How to capture the undertainties statistically**. There are plenty of undertainties associated with MOT. When a measurement track association is made, there is a chance the measurement comes from other tracks or clutter. When a track has no measurement association, there is a chance for missed detection. There are always the possibility of birth and death. Heuristically, these uncertainty can be registered via a counter system as presented in JPDA project. In that system, no action would be taken unless the counter has reached the predetermined threshold.
In statistical filter, this uncertainty is registered via weights or existence probabilities. 
2. **Exhaustively enumerate all the scenarios**. This is most pertinent to PMBM and JPDA, specifically, Global Hypothesis and list of association matrices. Gobal constraints that can only be expressed via exhaustively enumerated possibilities. In the MOT case, the global constrain is that under the point target assumption (one target can only genenrate one measurement, this assumption is relaxed later under the extended measurement model) only one association pair is valid. 
