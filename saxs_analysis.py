# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:53:06 2023

@author: milsomay
"""


import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# SAXS/WAXS nxs reader class

class DiamondProcessedNxsReader():
    '''
    A class that reads processed nxs (h5) files collected from Diamond Light Source
    (I22). 
    
    You may need to change the hdf_file_locations dict to where the locations for
    I (intensity), q and I_err are in the nxs file, if this changes. 
    
    Can access intensity --> DiamondProcessedNxsReader.I (either 1D or 2D)
    Can access q --> DiamondProcessedNxsReader.q
    Can access I_err --> DiamondProcessedNxsReader.I_err
    
    '''
    
    def __init__(self,processed_filename,raw_filename=None,
                 hdf_file_locations = {'I':'processed/result/data',
                                   'q':'processed/result/q',
                                   'I_err':'processed/result/errors',
                                   'pipeline':None,}):
        
        self.filename = processed_filename
        
        self.I = None
        self.q = None
        self.I_err = None
        
        self.It = None # not used but can add to class definition if path in nxs file known
        self.I0 = None # not used but can add to class definition if path in nxs file known
        
        self.title = None # not used but can add to class definition if path in nxs file known
        
        self.pipeline = None # not used but can add to class definition if path in nxs file known
        
        self.hdf_file_locations = hdf_file_locations
        
        # search_terms = ['data','q','errors']
        
        collected_attr = [self.I,self.q,self.I_err]
        
        with h5.File(processed_filename,'r') as f:
            self.I = np.copy(f.get(self.hdf_file_locations['I']))
            self.I_array = np.copy(f.get(self.hdf_file_locations['I']))
            self.q = np.copy(f.get(self.hdf_file_locations['q']))
            self.I_err = np.copy(f.get(self.hdf_file_locations['I_err']))
            
            # COULD ADD self.title; self.pipeline; self.I0, self.It - if path known
        
        # optional transformed I dependent on if 2-, 3- or 4-D data
        # i.e. a grid, vertical/horizontal or single scan
        if len(self.I.shape) > 2:
            self.I = np.transpose(self.I[:,0,:])


def plot_heatmap_linescan(data,cmap='viridis',scan_step_size=None,
                          ):
    '''
    

    Parameters
    ----------
    data : DiamondProcessedNxsReader object
        
    cmap : str, optional
        The matplotlib colourmap to use. The default is 'viridis'.
        
    scan_step_size : float, optional
        The step size in mm for the line scan. The default is None. 

    Returns
    -------
    Plots a heatmap plot with Log normalised intensities. Figure object also returned.
    This is for a linescan.

    '''
    
    fig = plt.figure()

    y = np.linspace(0,len(data.I_array[:,0,0]),len(data.I_array[:,0,0])+1)
    
    # multiply the scan step number (y) by the step size to convert to y/x position
    # in the line scan
    if scan_step_size is not None:
        y *= scan_step_size
    
    # q data on the x axis (this could be in A-1 or nm-1 - CHECK THAT)
    x = np.append(0.0,data.q)
    
    plt.pcolormesh(x,
               y,
               data.I.T,norm = colors.LogNorm(),cmap=cmap)
    
    if scan_step_size is not None:
        plt.ylabel('Vertical position / mm')
    else:
        plt.ylabel('Scan number')
    
    plt.xlabel('q / nm$^{-1}$')
    
    plt.tight_layout()
    
    plt.show()
    
    return fig

    


if __name__ == '__main__':

    SCAN_NUMBER = 671452 # change this to look at a particluar scan
    
    # change this to where the processed nxs files are
    # filenames might be named slightly differently
    filepath = f'processed/i22-{SCAN_NUMBER}_saxs_Transmission_IvsQ_processed.nxs'
    
    
    data = DiamondProcessedNxsReader(filepath)
    
    # BELOW WILL PLOT A HEATMAP FOR A VERTICAL/HORIZONTAL LINE SCAN
    
    fig = plot_heatmap_linescan(data,scan_step_size=0.02)
    
    # use below to save the heatmap figure if desired
    FIG_TITLE = f'{SCAN_NUMBER}_OA_SO_Fru_1to1to1_220min_O3.png'
    plt.figure(fig.number)
    plt.savefig(FIG_TITLE,dpi=300)
    print('\n***********************')
    print(f'SAVED FIGURE AS: {FIG_TITLE}')
    print('***********************')
    
    # BELOW IS AN EXAMPLE IF YOU WANT TO JUST PLOT A SINGLE PATTERN (IF NXS ONLY CONTAINS A SINGLE PATTERN)
    # (UNCOMMENT BELOW)
    
    # plt.figure()
    # plt.plot(data.q, data.I)
    
    # plt.ylabel('Intensity')
    # plt.xlabel('q / [q units - check]')
    
    # plt.tight_layout()
    # plt.show()
    
    # BELOW IS AN EXAMPLE OF HOW TO PLOT A SPECIFIC FRAME FROM A LINESCAN (IF NXS IS A LINESCAN)
    # UNCOMMENT BELOW
    
    # plt.figure()
    # plt.plot(data.q, data.I[:,1]) # plotting all of the 2nd frame (frame 1 - python counts from 0)
    
    # plt.ylabel('Intensity')
    # plt.xlabel('q / [q units - check]')
    
    # plt.tight_layout()
    # plt.show()
    
    
    
    
       