'''
PDA and JPDA are the data association schemes. Specifically:
i) PDA algorithm outline, as presented on page 10 in [1]:
    - a) A PDAF has a selection procedure for the validated measurements at the current time. 
    - b) For each such measurement, an association probability is computed for use as the weighting of this measurement in the combined innovation. 
            The resulting combined innovation is used in the update of the state estimate; this computation conforms to property P2 of the pure 
            MMSE estimator even though P2 is conditioned on satisfying P1 exactly;  nevertheless, P2 is still used for the sake of simplicity when P1 is satisfied approximately. 
    - c) The  final  updated  state  covariance  accounts  for  the  measurement origin uncertainty.The stages of the algorithm are presented next. 
ii) JPDA algorithm outline, as presented on page 11 in [1]:
    - a) The  measurement-to-target  association  probabilities are computed jointly across the targets. 
    - b) In view of the assumption that a sufficient statistic is available,  the  association  probabilities  are  computed  only for the latest set of measurements. This approach conforms to the results from the section “The Optimal Estimator for the Pure MMSE Approach.” 
    - c) The state estimation is done either separately for each target  as  in  PDAF,  that  is,  in  a  decoupled  manner,  resulting  in  JPDAF,  or  in  a  coupled  manner  using  a  stacked state vector, resulting in JPDACF
'''

import numpy as np
from utils import gauss, compute_marginal_probability
birth_initiation = 3

"For one constant velocity target, perform PDA once per frame."
class CVTarget:
    def __init__(self,
                 x: np.float,
                 y: np.float,
                 F: np.array,
                 H: np.array,
                 Q: np.array,
                 R: np.array,
                 eta: np.float,
                 P_D: np.float,
                 lambda_clutter: np.float):
        # lambda_clutter is spatial density of clutter under Poisson clutter model(Thus in this code we use parametric PDA, see equation (37) in [1].).
        self.lambda_clutter = lambda_clutter    # self.lambda_clutter is spatial density of false measurements(clutter).
        self.target_state = np.array([
            [x],
            [y],
            [0],
            [0],
        ])                                     # self.target_state represents the state of each target.
        self.F = F                                          # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
        self.H = H                                          # H is measurement matrix which represents the measurement model in update step.
        self.Q = Q                                          # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R                                          # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.P = np.eye(4, 4)                               # State covariance matrix, P, is initialized using 4-by-4 diagonal matrix for the current target
        self.S = self.innov_cov(self.H, self.P, self.R)     # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        self.eta = eta
        
        self.P_D = P_D  # P_D is the target detection probability.
        self.birth_counter = birth_initiation   # when the target is initiated, counter is set at one                                 
        self.death_counter = 0   # initiate a death counter
        self.measurements_within_gating = []
        self.likelihood_probability_for_each_measurement = {} # a dictionary where the key is the position of each measurement and value its likelihood as seen from this target

    @staticmethod
    def innov_cov(H, P, R):
        # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        return H@P@np.transpose(H) + R

    # Kalman prediction. 
    def predict(self):
        # Calculate all the variables regarding motion model.
        self.target_state = self.F@self.target_state            # Equation (30) in [2].
        self.P = self.F@self.P@np.transpose(self.F) + self.Q    # Equation (32) in [2].
        self.S = self.innov_cov(self.H, self.P, self.R)         # Equation (33) in [2].

    # Kalman update with JPDA.
    def jpda_update(self, marginal_probability, Z: list):
        '''self.S = self.innov_cov(self.H, self.P, self.R) # update self.S based on the new self.P. This step has been accomplished by calling predict. Redundant'''
        zpred = self.zpred() # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = [] # innovation is defined by equation (2.5) in [2]
        betas = []
        t_position = (self.target_state[0][0], self.target_state[1][0])     # Position of current target.
        for z in Z: 
            if z.size != 2:
                raise Exception("z has wrong dimension", z)  # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
            innovations.append(z - zpred) # See eqation (2.14) in [2]
            m_position = (z[0][0], z[1][0])     # Position of current measurement which is in the gating region of current target.
            # Obtain the marginalized association probability between target t and measurement j (in the gating area of target t), beta_j^t with j = 0, 1, 2, ...
            betas.append(marginal_probability[t_position, m_position])
         
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association".
        # According to equation. 3.20 in [2].
        beta_0 = 1 - np.sum(betas)

        # Reduce the mixture into a combined innovation weighted by betas(association probabilities).
        combined_innovation = np.zeros_like(zpred)
        for j, innovation_for_each_measurement in enumerate(innovations):
            combined_innovation += betas[j] * innovation_for_each_measurement               # Calculate the combined innovation, v(k). See equation (40) in [1].
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)   # Calculate the Kalman gain, W(k). See equation (41) in [1].
        self.target_state = self.target_state + W@combined_innovation            # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].

        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j]*innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation@combined_innovation.T)@W.T

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov

    def pda_update(self, Z: list):
        """ self.S = self.innov_cov(self.H, self.P, self.R)         # I don't understand why we need it again here! """
        zpred = self.zpred()        # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = []
        betas_unorm = []
        for z in Z:
            if z.size != 2:
                # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
                raise Exception("z has wrong dimension", z)
            # Calculate innovation for each measuremnt, (z - zpred), which is v_i(k) in equation (40) in [1].
            innovations.append(z - zpred)
            # TODO: The calculation of likelihood ratio of measurement and beta are wrong! Need to redo this part by following the equation (37) and (38) in [1] exactly!
            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) in [1].
            betas_unorm.append(self.P_D * gauss(z, zpred, self.S))
        # Calculate the probablity of "association between every measurement(in the gating area, if gating is performed before update) and current target is the correct association", beta_i(k).
        # See equation (37) in [1].
        betas = betas_unorm / np.sum(betas_unorm)

        # Reduce the mixture into a combined innovation weighted by betas(association probabilities).
        combined_innovation = np.zeros_like(zpred)
        for j, innovation_for_each_measurement in enumerate(innovations):
            # Calculate the combined innovation, v(k). See equation (40) in [1].
            combined_innovation += betas[j] * innovation_for_each_measurement
        # Calculate the Kalman gain, W(k). See equation (41) in [1].
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)

        # Calculate the updated state estimation, x^hat(k|k). See equation (39) in [1].
        self.target_state = self.target_state + W@combined_innovation

        # Calculate the spread of the innovation, P^hat(k). See equation (44) in [1].
        beta_boi = 0
        for j, innovation_for_each_measurement in enumerate(innovations):
            beta_boi += betas[j] * \
                innovation_for_each_measurement@innovation_for_each_measurement.T
        sprd_innov = W@(beta_boi - combined_innovation @
                        combined_innovation.T)@W.T

        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association".
        beta_0 = self.lambda_clutter * (1 - self.P_D)

        # Calculate the updated state covariance matrix, P(k|k), according to combination of equation (42),(43) in [1].
        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov

    def zpred(self):
        # Predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        zpred = self.H@self.target_state
        #zpred = np.array([[zpred[0][0]], [zpred[1][0]]])
        return zpred
          
    def gating(self, Z):
        zpred = self.zpred()  # Calculate predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        # Only keep the measurements within the gating area for currect target, according to equation (34) in [1].
        gated_measurements = np.array([z for z in Z if np.transpose(z - zpred)@np.linalg.inv(self.S)@(z - zpred) < self.eta])
        self.measurements_within_gating = gated_measurements
        return gated_measurements.reshape((gated_measurements.size // 2, 2, 1))

    def compute_likelihood_probability_for_each_measurement(self, measurements):
        '''
        Generate a dictionary whose key is the measurement and the corresponding value is its likelihood, See equation (38) or (48) in [1].
        Seeing from this particular target return a dictionary [measurement]: likelihood.
        '''
        for m in measurements:
            if m.size != 2:
                raise Exception("m has wrong dimension", m)
            mpred = self.zpred()
            # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) or (48) in [1].
            self.likelihood_probability_for_each_measurement[(m[0][0], m[1][0])] = self.P_D * gauss(m, mpred, self.S)

    def increase_birth_counter(self): # when a potential track has measurement associated, update counter
        self.birth_counter += 1
    def decrease_birth_counter(self):
        self.birth_counter -= 1
    def read_birth_counter(self): # return the birth counter
        return self.birth_counter
    def increase_death_counter(self): # when a potential death track lacks data association, update death counter
        self.death_counter +=1
    def decrease_death_counter(self):
        self.death_counter -=1
    def read_death_counter(self): #return death counter
        return self.death_counter
      
    def clear_measurements_association(self): # reset measurements association
        self.measurements_within_gating = []

    def read_measurements_within_gating(self): # return the associated measurements
        return self.measurements_within_gating
    
    def read_probability_for_each_measurement(self):
        return self.likelihood_probability_for_each_measurement