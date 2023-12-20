#  baryon-sweep

[![DOI](https://zenodo.org/badge/695291842.svg)](https://zenodo.org/badge/latestdoi/695291842)

This code represents a custom outlier rejection algorithm for JWST/NIRSpec IFS data, described in Hutchison et al., submitted to PASP (arxiv:####.#####).  The algorithm has been split into different jupyter notebooks in order to make it easier to parse and easier to test/troubleshoot specific sections for your own data.


### Order of Operations
1. `generating-target-mask.ipynb`, creates the 2D pixel mask and mask layers (Section 2.1 in H+)
1. `layered-sigma-clipping.PART1.ipynb`, processes the sky (non-science target) spaxels (Section 2.2.1 in H+)
1. `layered-sigma-clipping.PART2.ipynb`, processes the science target spaxels (Section 2.2.2 in H+)
1. `layered-sigma-clipping.PART3.ipynb`, combines into one final post-processed cube ready for science (Section 2.2.3 in H+)


### Other Files in Repository
Here we describe the other files in this repository, some that are optional and some that are necessary.

- `science-target.txt`, a dictionary describing the location of the level 3 IFU cube from the jwst pipeline and other information relevant to the science target such as the target name, redshift, grating of observation, etc.  Will be read into each of the notebooks.
- `routines.py`, a series of helper functions made to help keep the notebooks as clean and user-friendly as possible.
- `troubleshooting_visually.py`, as series of plotting functions that can be used to inspect the IFU wavelength slices and the individual spaxel spectra, comparing before & after the algorithm.



>[!NOTE]  
>Near-future plans will include sharing a more detailed example of how I make & modify the mask layers for a specific science target.  However, for now the material in the `generating-target-mask.ipynb` notebook should be enough to get started!

