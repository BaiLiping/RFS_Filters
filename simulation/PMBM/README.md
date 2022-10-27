## PMBM Filter Procedure(Copied from https://github.com/neer201/Multi-Object-Tracking-for-Automotive-Systems-in-python/tree/main/src/mot/trackers/multiple_object_trackers/PMBM)

1. Prediction of Bernoulli component.
2. Misdetection update of Bernoulli component.
3. Object detection update of Bernoulli component.
4. Prediction of Poisson Point Process (PPP).
   1. Predict Poisson intensity for pre-existing objects.
   2. Incorporate Poisson birth intensity into Poisson intensity for pre-existing objects.
5. Misdetection update of PPP.
6. Object detection update of PPP.
   1. For each mixture component in the PPP intensity, perform Kalman update and calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
   2. Perform Gaussian moment matching for the updated object state densities resulted from being updated by the same detection.
   3. The returned likelihood should be the sum of the predicted likelihoods calculated for each mixture component in the PPP intensity and the clutter intensity. (You can make use of the normalizeLogWeights function to achieve this.)
   4. The returned existence probability of the Bernoulli component is the ratio between the sum of the predicted likelihoods and the returned likelihood. (Be careful that the returned existence probability is in decimal scale while the likelihoods you calculated beforehand are in logarithmic scale.)
7. PMBM prediction.
8. PMBM update.
   1. Perform ellipsoidal gating for each Bernoulli state density and each mixture component in the PPP intensity.
   2. Bernoulli update. For each Bernoulli state density, create a misdetection hypothesis (Bernoulli component), and m object detection hypothesis (Bernoulli component), where m is the number of detections inside the ellipsoidal gate of the given state density.
   3. Update PPP with detections. Note that for detections that are not inside the gate of undetected objects, create dummy Bernoulli components with existence probability r = 0; in this case, the corresponding likelihood is simply the clutter intensity.
   4. For each global hypothesis, construct the corresponding cost matrix and use Murty's algorithm to obtain the M best global hypothesis with highest weights. Note that for detections that are only inside the gate of undetected objects, they do not need to be taken into account when forming the cost matrix.
   5. Update PPP intensity with misdetection.
   6. Update the global hypothesis look-up table.
   7. Prune global hypotheses with small weights and cap the number. (Reduction step ?)
   8. Prune local hypotheses (or hypothesis trees) that do not appear in the maintained global hypotheses, and re-index the global hypothesis look-up table. (Reduction step ?) 
9. Reduction
   1.  Prune the Poisson part by discarding components whose weight is below a threshold.
   2.  Prune global hypotheses by keeping the highest Nh global hypotheses.
   3.  Remove Bernoulli components whose existence probability is below a threshold or do not appear in the pruned global hypotheses
10. Object states extraction.
    1.  Find the multi-Bernoulli with the highest weight.
    2.  Extract the mean of the object state density from Bernoulli components with probability of existence no less than a threshold.
