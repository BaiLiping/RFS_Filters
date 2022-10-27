import numpy as np

from utils import gauss


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
        ])                                                  # self.target_state represents the state of each target.
        self.F = F                                          # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in prediction step.
        self.H = H                                          # H is measurement matrix which represents the measurement model in update step.
        self.Q = Q                                          # Motion noise(process noise) covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R                                          # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.P = np.eye(4, 4)                               # State covariance matrix, P, is initialized using 4-by-4 diagonal matrix for the current target
        self.S = self.innov_cov(self.H, self.P, self.R)     # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        self.eta = eta
        self.P_D = P_D                                      # P_D is the target detection probability.

    @staticmethod
    def innov_cov(H, P, R):
        # Innovation covariance matrix, S, equals to H(k)@P(k)@H(k)^T + R(k), where k is the time stamp(the k^th frame). See equation (33) in [1].
        return H@P@np.transpose(H) + R

    # Kalman prediction. 
    def predict(self):
        # Calculate all the variables regarding motion model.
        self.target_state = self.F@self.target_state            # Equation (30) in [1].
        self.P = self.F@self.P@np.transpose(self.F) + self.Q    # Equation (32) in [1].
        self.S = self.innov_cov(self.H, self.P, self.R)         # Equation (33) in [1].

    # Kalman update with PDA.
    def pda_and_update(self, Z: list):
        """ self.S = self.innov_cov(self.H, self.P, self.R)         # I don't understand why we need it again here! """
        zpred = self.zpred()        # Calculate predicted measurement, z^hat(k|k-1). Equation (31) in [1], which is actually part of prediction step.
        innovations = []
        betas_unorm = []
        for z in Z:
            if z.size != 2:
                raise Exception("z has wrong dimension", z)         # Since z should always has measurement information of (x, y), thus dimension needs to be as 2.
            innovations.append(z - zpred)                           # Calculate innovation for each measuremnt, (z - zpred), which is v_i(k) in equation (40) in [1].
            betas_unorm.append(self.P_D * gauss(z, zpred, self.S)/self.lambda_clutter)  # Calculate likelihood ratio of measurement z_i(k) originating from the current target, L_i(k). See equation (38) in [1].
        # Calculate the probablity of "association between every measurement(in the gating area, if gating is performed before update) and current target is the correct association", beta_i(k).
        # See equation (37) in [1]. 
        betas = betas_unorm / (np.sum(betas_unorm) + (1 - self.P_D))
        # Calculate beta_0, which is the probability of "none of the measurements associate to the current target is the correct association". See equation (37) in [1]. 
        beta_0 = (1 - self.P_D) / (np.sum(betas_unorm) + (1 - self.P_D))

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

    def zpred(self):
        # Predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        return self.H@self.target_state     
    
    def gating(self, Z):
        zpred = self.zpred()  # Calculate predicted measurement, z^hat(k|k-1), equals to H(k)@x^hat(k|k-1). Equation (31) in [1].
        # Only keep the measurements within the gating area for currect target, according to equation (34) in [1].
        gated_measurements = np.array([z for z in Z if np.transpose(z - zpred)@np.linalg.inv(self.S)@(z - zpred) < self.eta])
        return gated_measurements.reshape((gated_measurements.size // 2, 2, 1))
