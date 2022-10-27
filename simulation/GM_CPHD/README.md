# Background Understsanding of GM-CPHD Filter:

## Reference Papers for PHD:
- [1] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal Processing
- [2] 2003 R. Mahler "Multitarget Bayes Filtering via First-Order Multitarget
Moments" IEEE Transactions on Aerospace and Electronic Systems

## Lecture Series for PHD:
- https://youtube.com/playlist?list=PLadnyz93xCLhFinI8NO30-1e6SwCGRTIM

For exact implementation of GM-PHD Filter, one should refer to Table I, II, III of [1]. For the evolution of PHD, the concepts behind this set of methods, one should refer to [2] and R. Mahler's works in general.

## From Bayesian to GM-PHD:
**Bayesian Filters** propogate the entire distribution, yet it is computationally impossible for any real use cases.
**Kalman Filters** are Baysian Filter under the linear guassian assumption. It propogates first & second moments of a distribution. The second moments computation is usually quite evolved and can be intractable when the terms are large enough.
**Constant Kalman Filter** When the SNR is large enough and the variance/distance ratio is reasonable, first moment along is good enough of a mathematical character for a distribution (refer to [2] behind this reasoning). 
**PHD** Mathematically speaking, PHD is the first moment of a RFS distribution, yet the physical intuition behind it is not probability. It is called intensity because if you integrate PHD over a given area, you will get the expected cardinality of that area. PHD filters are multi-object filters analogous to Constant kalman Filter, since only the first moment of RFS distribution are propogated.
**GM-PHD** Intensity functions are modelled by Gaussian Mixture. The math behind the update steps of a poisson mixture bernoulli  is very evolved. But the conclusion is this: instead of association measurements to tracks, the entire update step becomes a embedded for loops:
```python
peudocode for update step of GM-PHD filter:
    for m in measurements:
        for guassian in all_guassian_components:
            add a new mean
            add a new covariance
            add a new weight
```
As we can see, there is no association whatsoever.Every measurement is use to update the parameters of all the guassians. This is why GM-PHD filters gained tractions. While the math behind it is evolved, the outcome is simply kalman filter steps for each guassian distribution.

## Reference Paper on GM-CPHD
- [1] 2006 B.-N. Vo, Anthonio Cantoni "The Cardinalized Probability Hypothesis Density Filter for Linear Gaussian Multi-Target Models", 
2006 Annual Conference on Information Sciences and Systems (CISS)
- [2] 2007 R. Mahler "PHD filters of higher order in target number" IEEE Transactions on Aerospace and Electronic Systems

## From GM-PHD to GM-CPHD
While GM-PHD filter successfully converting the data association steps into two for loops, the state extraction step is just wishy-washy. It is done by set a threshold for the intensity, and only keep the guassian distributions whose mean is greater than that threshold. How should that threshold be set? Why would the threshold be set that way? Those are all left to be decided by the people who implement the filter.

GM-CPHD filter make the final state extraction step a more precise one. It propogates the cardinality distribution alongside the GM-PHD filter, and the cardinality information is used in both state update step and the state extraction step.

## What is the cardinality distribution
As mentioned before, the first moment of RSF distribution is a function where if you integrate over a given area, you will get the expected cardinality. Cardinality distribution is the integral of intensity. 

The cardinality consists of two sets: birth set and surviving set. In [1], there is actually a spawning set, but in practice we just ignore that part and only concern ourselves with birth set and surviving set.

The birth is modelled as Poisson Point Process(PPP), and the survival is modelled as Poisson Mixture Bernoulli(PMB). The RFS distribution of the convolution of PPP birth distribution and PMB survive distribution. Therefore, the cardinality distribution is the convolution between cardinality_birth and cardinality_survive.

### Birth
The birth is modelled as a PPP, and the intensity of the PPP RFS distribution is modelled as GM. The cardinality_birth is the set integral of intensity and it is a poisson. The exact derivation can be found at the lecture series.

### Death/Survive
Instead of modelling death, the theory models survival instead. The survival is modlled as a PMB. The exact derivation will be skilled. We opt to follow Gasia's implementation of the cardinality_survive. The reasoning behind cardinality_survive is similar to that of cardinality_birth, derived by doing a set integral of intensity of survival RFS distribution.







