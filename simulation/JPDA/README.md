Overview of Tracking:
point target model based multi-object tracking 难点有两个：1. 在 cardinality（number of targets）的确立  2. 以及在point object这个假设下对clutter&missed detection的处理。

random vector based Bayesian filter(即你写的data association filter, e.g. GNN, JPDA, MHT)的方案只能解决问题2，解决不了问题1, 需要假设问题1已经知道答案了。那基于RFS based Bayesian filter(PHD, LMB, PMBM, etc)可以同时解决问题1和2，其中PHD是不做显示的data association hyphotheses处理的，LMB和PMBM则是把所有可能的data association hyphotheses穷尽之后通过各种pruning和merge的方式减少hyphothese的数量。

The Problem JDPA tries to solve:

with respect to any given target, measurements  from interfering targets  do  not behave at all like the random (Poisson) clutter assumed above. Rather,  the  probability density of each candidate measurement  must  be  computed based upon  the densities of all targets that  are close enough (when projected  into  the measurement space) to interference.

The  joint probabilistic  data  association (JPDA) and PDA approaches utilize the same estimation  equa- tions;  the difference is  in the way the association  probabilities are computed. Whereas the PDA algorithm computes pit, j = 0, 1, -0, m, separately for  each t, under  the assumption that all measurements not associated with target t are false (i.e., Poisson-distributed  clutter),  the JPDA  algorithm computes pi’ jointly across the  set  of T targets and  clutter.  From  the  point of view of  any target, this accounts for false measurements from  both discrete interfering  sources (other targets) and ran- dom  clutter.