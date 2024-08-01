# flux-cal
Photometric Calibration of SuperBIT Telescope

Liam O'Shaughnessy (Princeton '26, physics) under the direction of Dr. William Jones - Princeton University - Department of Physics - Summer 2024

In this project, my objective was to produce a lower-order flux calibration for the SuperBIT telescope with respect to a reference (the Subaru Telescope Hyper-Suprime Cam).


Two VERY important notes:

First, do not immediately trust plots you see as they may be dated as a new technique might have been employed. The report and slides have definitive plots. If you want one of the plots, it would be best to RERUN the "flux_plotter.py" script!!
Second, the "fits_files" directory is empty for cosmosk - oh no! This is due to the sizes of the .fits files being very large. This is not a problem, however, since information about objects is available in the "catalogs" or "cat_files" directories. This would only impact visualization through a software such as DS9, yet I believe that the SuperBIT data and HSC data are available online to download, if you would like to do this.


A bit of background: 

SuperBIT is a balloon-bound telescope that was flown in 2023 (this means it avoids heavy atmospheric interference while being lower-cost - hurray!).
The data received was of various regions of the sky, and this was registered by photons hitting a sensor that would count the number of hits.
The telescope had a number of different photon sensors (bands) that would each respond well to a certain wavelength of light.
When we put together a picture of the sky from this data, we can see objects, yet the pixel values in the image are not immediately interpretable - how many units of flux is one "BIT count"?
Hence, there needs to be a way to relate the units of the data we are getting to a more understandable unit of flux (e.g. janskys) - this is the calibration.


Idea of the project:

If one knew, from another telescope, how bright some celestial object was, and the same object was identified in SuperBIT, then we could estimate a conversion factor by (Reference brightness)/(SuperBIT brightness) - if we averaged that over a number of objects, we could get a robust conversion factor.
This must be done for every combination of bands from SuperBIT and the reference telescope, since these are the sources of flux data. However, the actual values of brightness that the telescopes read are a function of the product of source spectra (what wavelengths of light is the celestial object emitting) and band receptivity (what wavelengths of light can the particular sensor read well).
Assuming both telescopes are looking at the same object (so the spectra are the same), this single conversion factor can be wildly inaccurate if the band receptivities are very different.
Hence, it is good to calibrate similar bands, otherwise the approximation of the "single conversion factor" will likely be very off.
Detailed specifics can be found in the "SuperBIT_Report.pdf" file.


Repository Layout:

This layer contains reports and scripts (which needed to be able to see all files). "SuperBIT_Report.pdf" is a LaTEX report of the project with references to materials, and "SuperBIT Calibration.pptx" is slides from a later meeting. The "default" files are for SourceExtractor, which was used to get object parameters, while the various Python files would create catalogs of common objects, make intermediate plots, and do the bulk of the analysis ("flux_plotter.py"). The directories "a1689" and "cosmosk" are mostly identical (this is a function of simply changing paths when running the scripts). Inside, you can find files of extracted object parameters (.cat files from Source Extractor), commonly identified objects with parameters (.csv catalogs), and various plots.
