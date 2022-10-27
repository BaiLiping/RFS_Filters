import numpy as np

from cvtarget import CVTarget


"Constant Velocity Target Maker, it is just kind of wrapper to call CVTarget, which runs the core part of Kalman + JPDA."
class CVTargetMaker:
    F = np.empty((4, 4))
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
    Q = np.eye(4, 4)
    R = np.eye(2, 2)

    def __init__(self, T: np.float, Q: np.array, R: np.array, eta: np.float, P_D: np.float, lambda_clutter: np.float):
        self.lambda_clutter = lambda_clutter
        self.Q = Q              # Motion noise covariance matrix, Q, is initialized with 4-by-4 diagonal matrix(since motion is modeled by constant velocty model using x, y, x_v, y_v).
        self.R = R              # Measurement noise covariance matrix, R, is initialized with 2-by-2 diagonal matrix(since measurement model only measures x, y).
        self.F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])                      # F is the transition matrix with constant velocity motion model, which will be used to multiply with vector [x, y, x_v, y_v]^T in the prediction step.
        self.eta = eta
        self.P_D = P_D          # P_D is the target detection probability.

    def new(self, x: np.float, y: np.float):
        return CVTarget(x, y, self.F, self.H, self.Q, self.R, self.eta, self.P_D, self.lambda_clutter)
