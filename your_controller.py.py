# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
        
        self.psi_cum_err = 0
        self.vel_cum_err = 0
        self.psi_pre_err = 0
        self.vel_pre_err = 0
        
        
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
      
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
       
        _, closest_pt = closestNode(X, Y, trajectory)
        time_ahead = 15  # Ahead mechanism

        if (closest_pt + time_ahead >= trajectory.shape[0]):
            time_ahead = 0
            
        X_des = trajectory[closest_pt + time_ahead, 0]
        Y_des = trajectory[closest_pt + time_ahead, 1]
        psi_des = np.arctan2(Y_des - Y, X_des - X)    
       

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        
        psi_err = wrapToPi(psi_des - psi)
        psi_df_err = (psi_err - self.psi_pre_err) / delT
        self.psi_cum_err += psi_err * delT
        self.psi_pre_err = psi_err

        kp1 = 3.5
        ki1= 0.001
        kd1 = 0.001

        delta = kp1 * psi_err + ki1 * self.psi_cum_err + kd1 * psi_df_err


        if (delta < -3.1416 / 6):
            delta = -3.1415 / 6
            
        elif(delta > 3.1416 / 6):
            delta = 3.1416 / 6


        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        
        vel_err = (np.power(np.power(X_des - X, 2) + np.power(Y_des - Y, 2), 0.5)) / delT
        vel_df_err = (vel_err - self.vel_pre_err) / delT
        self.vel_cum_err += vel_err * delT
        self.vel_pre_err = vel_err

        kp2 = 10
        ki2 = 0.0001
        kd2 = 0.0001
        F = kp2 * vel_err + ki2 * self.vel_cum_err + kd2 * vel_df_err
        if (F > 15736):
            F = 15736
        elif(F < 1000):
            F = 1000
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
