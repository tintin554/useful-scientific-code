# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:16:38 2024

@author: adamm
"""

import numpy as np

from scipy.optimize import curve_fit, differential_evolution


def exponential_decay(x, a, b, c):
    """
    Exponential decay function.
    
    Args:
      x: Independent variable.
      a: Initial value.
      b: Decay constant.
      c: Asymptote. 
    
    Returns:
      Exponential decay function evaluated at x.
    """
    
    # return the y values at each x value for this exponential decay
    return a * np.exp(-b * x) + c

def fit_exponential_decay(xy_data, global_optimisation = False,
                          bounds = None,popsize=15,maxiter=1000):
    """
      Fit an exponential decay to x-y data.
    
      Args:
    xy_data: A 2-column numpy array with x-data in the 1st, y-data in the 2nd.
    
    global_optimisation: A boolean. True if global optimisation desired (differential evolution)
    
    bounds: a list of bounds (a tuple/list) for each argument (a, b, c) in that order.
            i.e. bounds = ([1e-1,1e-5,0.0], [10,1e-1,10]) --> first list is list of lower bounds; second list is list of upper bounds
            
    popsize: (for differential evolution); the number of parameter sets initiated.
            Total population size is popsize * number_of_varying_params (see SciPy docs).
            
    maxiter: maxiumum number of iterations for the differential evoltuion algorithm.
    
    
      Returns:
    Fitted parameters and their uncertainties.
    """
    x = xy_data[:,0]
    y = xy_data[:,1]
    
    
    # Define the objective function.
    def objective(params, x, y):
        '''
        This takes in a list of the varying parameters and calculates 
        the cost function between the model output (predicted_y) and the experimental
        data (y).

        Parameters
        ----------
        params : A list of varying parameters given to the exponential function.
        x : x-data to fit with.
        y : y-data to fit to.

        Returns
        -------
        The value of the cost function (mean squared error).

        '''
        a, b, c = params
        predicted_y = exponential_decay(x, a, b, c)
        
        # the cost function is 
        return np.mean((predicted_y - y)**2)
    
    # Perform the fit.
    if global_optimisation:
        #Define the bounds for the fitted parameters.
        bounds_DE = []
        for i in range(len(bounds[0])):
            lb, ub = bounds[0][i], bounds[1][i]
            bounds_DE.append((lb,ub))
        #print(len(bounds_DE))
        # Perform the fit.
        result = differential_evolution(objective, bounds_DE, args=(x,y),
                                        popsize=popsize,maxiter=maxiter)
        
        # Return the fitted parameters and their uncertainties.
        params = result.x
        uncertainties = np.zeros(len(params)) # can't get from this
        print('Success: ', result.success)
    else: 
        if bounds is not None:
            params, cov = curve_fit(exponential_decay, x, y,bounds=bounds)
        else:
            params, cov = curve_fit(exponential_decay, x, y)
          
        # Calculate the uncertainties.
        uncertainties = np.sqrt(np.diag(cov))
      
    return params, uncertainties


# BELOW IS AN EXAMPLE USE WITH A 2-COLUMN DATAFILE

FILENAME = '' # type the filename (with extention)

xy_data = np.genfromtext(FILENAME) # look at numpy docs for different delimiters etc.

lower_bounds = [1e-1,1e-5,0.0]
upper_bounds = [10,1e-1,10]

bounds = (lower_bounds, upper_bounds)

# global optimisation with a popsize of 50 and maxiterations of 2000
fitted_params, uncerts = fit_exponential_decay(xy_data, global_optimisation=True,
                                                       bounds=bounds,
                                                       popsize=50,
                                                       maxiter=2000)

