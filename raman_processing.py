# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:41:34 2024

@author: adamm
"""


import numpy as np


import matplotlib.pyplot as plt
from matplotlib import rcParams as rc

rc['font.sans-serif'] = 'Arial'
rc['font.size'] = 14

from scipy import sparse
from scipy.sparse.linalg import spsolve

class RamanExperimentReader():
    '''
    A class that will read Raman spectra either as a series of columns (1st column Raman shifts, rest intensities)
    or as 2 columns with spectra stacked on top of each other.
    
    Inputs:
    -------
    
    filename: the filename
    laser_wavelength: laser wavelength (not used currently but might be needed to convert to Raman shift)
    skip_row: number of rows to skip if file has information included in the first n rows of the file
    '''
    def __init__(self,filename,laser_wavelength=532.0, skip_row=0):
        
        self.filename = filename
        
        self.raw_data = np.genfromtxt(filename,delimiter=',',skip_header=skip_row)
        
        self.laser_wavelength = laser_wavelength
        
        self.background_sub_spectra = []
        self.background_polynomials = []
        self.backgrounds = []
        
        # split into separate raman spectra
        self.n_spectra = 1
        
        self.spectra = []
        
        n_row, n_col = self.raw_data.shape
        
        if n_col > 3:
            for i in range(n_col):
                if i != 0:
                    spectrum = np.column_stack((self.raw_data[:,0], self.raw_data[:,i]))
                    self.spectra.append(spectrum)
                    self.n_spectra += 1
        else:
            spec_start_index = 0
            for i in range(len(self.raw_data[:,0])):
                if i != 0:
                    val = self.raw_data[i,0]
                    
                    if val < self.raw_data[i-1,0]:
                        self.n_spectra += 1
                        raman_spectrum = np.column_stack((self.raw_data[spec_start_index:i,0],
                                                          self.raw_data[spec_start_index:i,-1]))
                        self.spectra.append(raman_spectrum)
                        spec_start_index = i+1
            
        if self.n_spectra == 1:
            self.spectra.append(np.column_stack((self.raw_data[:,0], self.raw_data[:,-1])))


# assymetric least squares baseline (got this off stack overflow)
def baseline_als(y, lam=1e6, p=0.05, niter=100):
    '''
    Returns a baseline fitted to a set of y values. Can adapt for "assymetry"
    and "smoothness".
    
    Parameters
    ----------
    y : array
        intensity data.
    lam : float, optional
        smoothness factor. The default is 1e6.
    p : float, optional
        symmetry factor. The default is 0.05.
    niter : int, optional
        number of iterations. The default is 100.

    Returns
    -------
    z : array
        baseline y values.

    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
        
    return z    

def get_area(spectrum,raman_shift_bounds):
    '''
    
    Uses the trapezoidal rule to get the area between two points on an x-y curve.

    '''
    lb, ub = raman_shift_bounds
    
    raman_shifts = spectrum[:,0]
    
    y_vals = spectrum[:,1]
    
    lb_ind = 0
    ub_ind = 0
    
    for i, val in enumerate(raman_shifts):
        
        if val >= lb and lb_ind == 0:
            lb_ind = i
            
        elif val >= ub and ub_ind == 0:
            ub_ind = i
    
    # using the trapizoidal rule to get area under a curve
    area = np.trapz(y_vals[lb_ind:ub_ind+1], raman_shifts[lb_ind:ub_ind+1])
    
    #print(lb_ind, ub_ind)
    
    return area

def plot_heatmap(list_of_spectra,vmin=0,vmax=None,timestep=1.0,
                 cmap='viridis', title='', save=False):
    
    # get Raman shifts/wavelengths
    raman_shifts = list_of_spectra[0][:,0]
    
    # stack list of spectra into one array
    
    for i, spec in enumerate(list_of_spectra):
        
        if i == 1:
            
            stacked_spectra = np.array(spec[:,1])
            
        elif i > 0:
            stacked_spectra = np.column_stack((stacked_spectra,spec[:,1]))
            
    
    _, n_spectra = stacked_spectra.shape
    
    spec_numbers = np.linspace(0, n_spectra, n_spectra)
    times = spec_numbers * timestep
    
    if vmax is None:
        vmax = np.max(stacked_spectra)
    
    plt.figure()
    plt.title(title)
    plt.pcolormesh(times, raman_shifts, stacked_spectra,
                   vmin=vmin,vmax=vmax,cmap=cmap)
    
    plt.xlabel('Time / s')
    plt.ylabel('Raman shift / cm$^{-1}$')
    plt.colorbar()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(title + '.png', dpi=300)
    
    plt.show()