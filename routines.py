'''

Series of helper functions for use in the layered sigma-clipping algorithm 
defined in Hutchison et al., submitted to PASP.

To read more, see arxiv.org/abs/2312.12518

Created by Dr. Taylor Hutchison, NASA GSFC,
on behalf of the TEMPLATES team.

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import warnings
import numpy as np

from astropy import units as u
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import pandas as pd
import astropy.io.fits as fits
import sys,json
import os



def get_target_info(target,grat='g140h'):
    '''
    Reading in the target file that has all the necessary info.
    This is set up so that one could theoretically have a long list of
    targets in the "science-target.txt" file and all they'd need to do
    is specify the target & grating.
    
    INPUTS:
    >> target  -----------  the name of the science target
    >> grat (opt)  -------  the grating of the specified observation,
                            optional unless >1 grating for science target
                       
    OUTPUTS:
    >> science_target ----  dictionary of info on science target
    >> path  -------------  path to the data to be read in
    >> grating  ----------  specified grating
    '''
    # reading in file that has all of the galaxy values
    with open('science-target.txt') as f:
        dictionary = f.read()

    # reconstructing dictionary
    targets = json.loads(dictionary)
    path = targets['path']

    # double-checking names are right for dictionary
    try: 
        science_target = targets[target] # getting galaxy info
        gratings = list(science_target['grating'].keys()) # listing gratings used
        
        # chosing grating for sources with >1 grating observed
        if len(gratings) > 1: grating = grat
        else: grating = gratings[0]
        
        return science_target,path,grating
    
    except KeyError: 
        print(f'The available targets are: {list(targets.keys())[1:]}') # skips the path variable
        sys.exit(0) # exiting script



def get_mask(target,array_2d=True):
    '''
    INPUTS:
    >> target  -------------  the name of the galaxy mask I want
    
    OUTPUTS:
    >> mask_layers  --------  mask layers of the science target,
                              where the first entry is the full mask
    >> mask_layers_info  ---  S/N range info of each mask layer,
                              where the first entry is the full mask
    '''
    
    filename = f'{target}-mask-layers.fits'
            
    mask_layers_info = np.loadtxt(f'{filename[:-5]}-info.txt',delimiter='\t')
    mask_layers = []

    # double-checking names are right for dictionary
    try:
        with fits.open(filename) as hdul:
            for i in range(len(hdul)):
                # map layer
                target_mask = hdul[i].data
    
                # makings a list of coordinates
                coordinates = list(zip(*np.where(target_mask == 1)))
    
                if array_2d == False: mask_layers.append(coordinates)
                else: mask_layers.append(target_mask)
    
        return mask_layers, mask_layers_info

    except:
        print("\nWrong layers file and/or file doesn't exist yet.",end='\n\n')
        sys.exit(0) # kills script
        
        
        
        

def convert_MJy_sr_to_MJy(spec):
    '''
    The spectra from the reduced data are in MJy/sr.
    Converting to MJy so they can be converted to cgs units.
    
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"
                       
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but in MJy    
    '''
    # taking nominal pixel area from FITS header for data
    pix_area = 2.35040007004737E-13 # in steradians
    
    # converting spectrum flam 
    spec['flam'] *= pix_area # MJy/sr --> MJy

    # converting spectrum error
    spec['flamerr'] *= pix_area # MJy/sr --> MJy
    
    return spec.copy()
    


def convert_MJy_cgs(spec):
    '''
    INPUTS:
    >> spec  --------  a pandas dataframe set up with columns
                       "wave", "flam", "ferr"; assumes wave is 
                       in microns and flam, ferr are in MJy
    OUTPUTS:
    >> spec ---------  the same pandas dataframe but in cgs
    '''
    # converting from MJy/sr to MJy
    spec = convert_MJy_sr_to_MJy(spec.copy()) 
    
    # converting spectrum flux density to flam cgs units
    spec['flam'] *= 1e6 # MJy --> Jy
    spec['flam'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['flam'] *= 2.998e18 / (spec.wave.values*1e4)**2 # fnu --> flam
    
    # converting spectrum error to flam cgs units
    spec['flamerr'] *= 1e6 # MJy --> Jy
    spec['flamerr'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spec['flamerr'] *= 2.998e18 / (spec.wave.values*1e4)**2 # fnu --> flam
    
    return spec.copy()




def get_spec(x,y,d,e,h):
    '''
    INPUTS:
    >> x,y  ---------  pixel coordinates for a given spaxel
    >> d  -----------  data cube
    >> e  -----------  uncertainty cube
    >> h  -----------  header for data cube
    
    OUTPUTS:
    >> spec ---------  a pandas df spectrum, with columns of
                       wavelength, flux density, & assoc. error
    '''
    # making wavelength array
    wave = np.arange(h['CRVAL3'], 
                     h['CRVAL3']+(h['CDELT3']*len(d)), 
                     h['CDELT3'])
    
    # doing it this way to circumvent the Big Endian pandas error
    dat = [float(f) for f in d[:,int(y),int(x)].copy()]
    err = [float(f) for f in e[:,int(y),int(x)].copy()]

    spec = pd.DataFrame({'wave':wave,'flam':dat,'flamerr':err})
    spec = convert_MJy_cgs(spec.copy()) # converting to flam cgs
    return spec.copy()




def get_benchmark_wave_index(datacube,sli,h):
    '''
    INPUTS:
    >> datacube  ------  data cube
    >> sli  -----------  benchmark slice index
    >> h  -------------  header for data cube
    
    OUTPUTS:
    >> wave_mask ------  index of spectral window around benchmark slice
    '''
    
    # making wavelength array
    wave = np.arange(h['CRVAL3'], 
                     h['CRVAL3']+(h['CDELT3']*len(datacube)), 
                     h['CDELT3'])

    sli_wave = wave[sli] # wavelength of benchmark slice

    # defining spectral window based on grating
    if len(wave) > 2500: w_win = 0.0085 # microns, high res
    elif len(wave) > 1100: w_win = 0.05 # microns, medium res
    else: w_win = 0.1 # microns, prism

    # pulling out specified wavelength slices in cube, centered on benchmark slice
    wavemin,wavemax = sli_wave-w_win, sli_wave+w_win
    wave_mask = np.where((wave<wavemax)&(wave>wavemin),wave,-1)
    wave_mask = np.arange(len(wave_mask))[wave_mask > 0]
    return wave_mask
    
    
    
def get_yaxis_scaling(flam):
    '''
    Purely for aesthetics in plotting, yall know I love that
    
    INPUTS:
    >> flam  --------  flux density, 1D array
    
    OUTPUTS:
    >> scale  -------  the 10^N scale to divide the yaxis by in plotting
    '''
    median = np.nanmedian(flam) # initial scaling value, ___x10^N
    exponent = int(np.log10(median)) # getting just the exponent part, N
    scale = np.power(10.,exponent) # 1x10^N
    return scale






