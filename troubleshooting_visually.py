'''

These are plotting helper functions in case you want or need to explore the data, to investigate a certain slice or spaxel's spectrum.  Personally, I found some of these most useful in the following ways:

1. checking different slices to see how the clipping process worked (before & after)
2. looking at individual spaxels (before & after)

For the second reason, this was most useful for me when looking at spaxels around the edge of the IFU field of view, where the S/N is lower due to being on/near the edge but the spaxel flux density values should be in higher S/N layers.  It's a niche case, but for the example galaxy shown in H+, submitted, this was necessary.


--------------------
As a final note, I code through scripting, so these functions were often run in my ipython terminal after running the algorithm scripts.  Since we converted those scripts to Jupyter notebooks (to make them more user-friendly), these functions were adapted to be importable.  HOWEVER, if you'd like, you can copy-paste the code itself into the bottom of the Jupyter notebook of choice -- then you can cut down the number of input variables needed (because the notebook will have most of them already in the memory).

To help with this, I've included a commented out version of what those inputs could be.
-------------------------


Created by Dr. Taylor Hutchison, NASA GSFC,
on behalf of the TEMPLATES team.

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import warnings
import numpy as np

from astropy import units as u
from astropy.stats import sigma_clip

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import astropy.io.fits as fits
import sys,json




def check_slice(sli,datacube,final_clipped,clipped_pixels):
    '''
    This function can be run after the final outputs are made in 
    each of the three "layered-sigma-clipping..." notebooks.  It 
    makes a plot that looks very similar to Fig 3 in H+, submitted.
    
    INPUTS:
    >> sli  ---------------  benchmark slice in IFU that you want to inspect
    >> datacube  ----------  the original IFU cube, from the pipeline
    >> final_clipped  -----  the post-processed IFU cube, from this algorithm
    >> clipped_pixels  ----  the log of which pixels are clipped in each IFU slice
                       
    OUTPUTS:
    plots a three-panel figure that shows 1) the specified IFU slice from
    the pipeline cube (before the algorithm), 2) the same slice from the
    post-processed cube (after the algortihm), and the same slice from the
    log of which pixels have been clipped in that slice.
    
    If this code is pasted into bottom of notebook, function can become 
    def check_slice(sli):
    '''
    plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,3,width_ratios=[1,1,1],wspace=0)

    ax = plt.subplot(gs[0]); ax.axis('off')
    ax.set_title('original pipeline slice')
    ax.imshow(datacube[sli],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
               cmap='viridis')

    ax = plt.subplot(gs[1]); ax.axis('off')
    ax.set_title('custom outlier rejection')
    ax.imshow(final_clipped[sli],clim=(-5e-3*pmap_scale,5e-2*pmap_scale),origin='lower',
               cmap='viridis')

    ax = plt.subplot(gs[2]); ax.axis('off')
    ax.set_title('pixels clipped in slice')
    # ax.imshow(full_mask,origin='lower',cmap='Greys',zorder=0,alpha=0.3)
    ax.imshow(clipped_pixels[sli],origin='lower',cmap='Blues',alpha=0.5)
    ax.text(0.047,0.927,'science target mask',color='grey',transform=ax.transAxes,fontsize=13)
    ax.text(0.047,0.87,'clipped pixel',color='C0',transform=ax.transAxes,fontsize=13,alpha=0.8)

    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    
    
def identify_spaxel(sli,cube,pix=False):
    '''
    This function was helpful in identifying certain spaxels that I
    wanted to inspect (for a given IFU wavelength slice).  Often I'd use
    this function to find the spaxel, then the "check_spec" function to
    inspect the spectrum.
    
    INPUTS:
    >> sli  -----------  benchmark slice in IFU that you want to inspect
    >> cube  ----------  the IFU cube of choice (any version)
    >> pix (opt) ------  a 1x2 array with x & y coordinates for a spaxel
                         (e.g., pix=[20,30]);  default False
                       
    OUTPUTS:
    plots a benchmark IFU slice, for exploration purposes.  When given x & y
    coordinates, the pix variable is plotted as a red scatterpoint on the 
    slice.
    
    If this code is pasted into bottom of notebook, function can become 
    def identify_spaxel(sli,pix=False):
    '''    
    plt.figure(figsize=(8,6))
    im = plt.imshow(cube[sli],origin='lower',clim=(-0.5,5),cmap='viridis')

    if pix != False:
        ax.scatter(pix[0],pix[1],s=40,edgecolor='k',color='r',lw=1.5)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    
    
    
def see_many_slices(sli,cube):
    '''
    This function is useful to check the extent of a specific artifact,
    across many IFU wavelength slices.  I used this to make sure that 
    the algorithm was properly treating each slice where the artifact/outlier
    was present.
    
    INPUTS:
    >> sli  -----------  benchmark slice in IFU that you want to inspect
    >> cube  ----------  the IFU cube of choice (any version)
                       
    OUTPUTS:
    plots five IFU wavelength slices, centered on the benchmark slice chosen.
    
    If this code is pasted into bottom of notebook, function can become 
    def see_many_slices(sli):
    '''
    plt.figure(figsize=(15,6.5))
    gs = gridspec.GridSpec(1,5,width_ratios=[1,1,1,1,1],wspace=0)

    for i in np.arange(-2,3): 
        ax = plt.subplot(gs[i+2]); ax.axis('off')

        im = ax.imshow(data_clipped[sli+i],clim=(-0.5,5),origin='lower',cmap='viridis')
        ax.text(0.05,0.9,f'slice: {sli+i}',fontsize=15,transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    
    
def check_spec(x,y,sli,datacube,final_clipped,h,ymax=False):
    '''
    This function is useful to check the extent of a specific artifact,
    across many IFU wavelength slices.  I used this to make sure that 
    the algorithm was properly treating each slice where the artifact/outlier
    was present.
    
    INPUTS:
    >> x,y  -------------  the coordinates of the spaxel of choice
    >> sli  -------------  benchmark slice in IFU that you want to inspect
    >> datacube  --------  the original IFU cube, from the pipeline
    >> final_clipped  ---  the post-processed IFU cube, from this algorithm
    >> h  ---------------  header for data cube
                       
    OUTPUTS:
    plots the 1D spectrum for the chosen spaxel, before & after the algorithm.
    
    If this code is pasted into bottom of notebook, function can become 
    def check_spec(x,y,sli,ymax=False):
    '''
    # pulling spectrum (setting errors to zero as we don't need that)
    cspec = get_spec(x,y,d=final_clipped,e=np.zeros(final_clipped.shape),h=h)
    ospec = get_spec(x,y,d=datacube,e=np.zeros(datacube.shape),h=h)

    w = cspec.loc[sli,'wave']

    plt.figure(figsize=(10,4))

    plt.step(ospec.wave,ospec.flam,where='mid',label='pipeline')
    plt.step(cspec.wave,cspec.flam,where='mid',label='clipped',alpha=0.7)

    plt.axvline(w,color='k')
    plt.legend()

    plt.xlabel('observed wavelength [microns]')
    plt.xlim(w-0.01,w+0.01)
    if ymax != False: plt.ylim(-5,ymax)
    else: plt.ylim(-5,35)

    plt.tight_layout()
    plt.show()
    plt.close('all')