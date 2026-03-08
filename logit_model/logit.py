import numpy as np
import matplotlib.pyplot as plt

class LogitModel:
    """
    This class gives the output for Logit Model, a 
    statistical tool for predicting the probability of
    something falling into one of two categories, like 'Yes' or 'No'
    """
    
    def __init__(self, features: float | list[float], weights: float | list[float]):
        self.features = features
        self.weights = weights

    def _score_calc(self):
        """Calculates the linear score (z)"""
        F = np.array(self.features)
        W = np.array(self.weights)
        z = np.dot(F,W)
        return z
    
    def sigmoid(self):
        """Converts a linear score into a probability between 0 and 1"""
        
        tol = 1e-6
        max = 1*10^6
        z = self._score_calc()
        numerator = 1
        denominator = 1 + np.pow(np.e, -z)
        
        if denominator < tol:
            S = 1
        elif denominator > max:
            S = 0
        else:
            S = numerator / denominator
        return S
    
    def plot_graphic(self):
        """Creates a graphic for probability of success with
        Sigmoid Function outputs"""
        plt.plot(self.features, self.sigmoid(), color='blue', label='Logit Curve')
        plt.ylabel('Probability of Success')
        plt.xlabel('Input Value (X)')
        plt.title('Logit Model Visualization')
        plt.legend()
        plt.show()


        