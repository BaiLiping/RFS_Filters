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


from PMBM_Filter_Point_Target import PMBM_Filter
from util import parse_args, gen_simulation_model, gen_filter_model, gen_simulation,filter_plot,plot_gospa, groud_truth_plot
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
    simulation_model = gen_simulation_model()
       
    # Initiate filter
    filter_model = gen_filter_model()
    Filter = PMBM_Filter(filter_model, args.Bayesian_filter_config, args.motion_model_type)
    
    for ith_simulation in range(args.number_of_monte_carlo_simulations):        
        # generate information for this simulation
        Z_k_all, targetStates_all, observations_all, clutter_all = gen_simulation(simulation_model,args.n_scan,args.simulation_scenario)

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
            
            '''
            STEP 1: Prediction 
            '''
            if i == 0:  # For the fisrt frame, there are only new birth targets rather than surviving targets thus we call seperate function.
                # the initial step the labmda for weight update is w_birthinit instead of w_birthsum
                filter_predicted = Filter.predict_initial_step()
            else:
                filter_predicted = Filter.predict(filter_pruned)
            '''
            STEP 2: Update 
            '''
            filter_updated = Filter.update(Z_k, filter_predicted, i) #Eq. 20 of [2]
            '''
            STEP 3: Extracting estimated states
            '''
            estimatedStates = Filter.extractStates(filter_updated)  # Extracting estimates from the updated intensity
            estimatedStates_mean = estimatedStates['mean']
            '''
            STEP 4: Pruning
            '''
            filter_pruned = Filter.prune(filter_updated)
            
            # Store Metrics for Plotting
            gospa,target_to_track_assigments,gospa_localization,gospa_missed,gospa_false = gp.calculate_gospa(targetStates, estimatedStates_mean, c=10.0 , p=2, alpha=2)
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
                filter_plot(fig,targetStates, observations, estimatedStates_mean, clutter,simulation_model)
         
        # stop timer for this simulation
        toc = time.process_time()
        if args.plot:
            plt.savefig('{}_tracker_overview.png'.format(args.simulation_scenario))
            plt.close(fig)
        # print out the processing time for this simulation
        print("This is the %dth monte carlo simulation, PMBM processing takes %f seconds" %(ith_simulation, (toc - tic)))
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