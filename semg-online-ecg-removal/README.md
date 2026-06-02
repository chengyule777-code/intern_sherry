# semg-online-ecg-removal

This package contains a Python algorithm for online filtering an sEMG signal using a stationary wavelet transform.
The class `sEMGOnlineFilter` in `sEMG_online_filter.py` can be used as an interface for using the online sEMG ECG removal algorithm in your project.
An example of how to do this is given in the file `sEMG_simulation_multichannel.py`. There you can see how to filter your data using the `sEMGOnlineFilter` class.
It is also shown, how to plot the raw and filtered signals fast. Therefore, a recorded respiratory sEMG with airway pressure measurement is used.

If you only want to use parts of the package, take a look at the given interface. There you can see how the individual methods can be used.

### Content
Python implementations of
- the online wavelet denoising algorithm (for single- and multichannel measurements)
- an online QRS detection
- online causal FIR and IIR filters
- a causal three-layer filter bank
- online calculation of the envelope of a signal
- an interface to easily filter a real-time measurement
- a method to identify the best ECG for QRS detection out of multiple given signals
- an example of how to use the interface

### Run
In order to start the example, you have to run `main.py`.

If you just want to use the provided interface, you only have to create an `sEMGOnlineFilter` and call `filter_sEMG_online(..)` on the object.
This interface is also used in the given example.

### Requirements
The requirements can be installed with
``
pip install -r requirements.txt
``
.

### Acknowledgements
This package contains a Python implementation of the Pan-Tompkins algorithm, which is a modified version of the Matlab implementation from the
[OSET](https://github.com/alphanumericslab/OSET) toolbox:  
R. Sameni, OSET: The open-source electrophysiological toolbox. Version 3.14, 2006-2023.

The wavelet denoising algorithm is based on the offline algorithm described in the following [publication](https://ieeexplore.ieee.org/document/8988257):  
E. Petersen, J. Sauer, J. Graßhoff and P. Rostalski, "Removing Cardiac Artifacts From Single-Channel Respiratory Electromyograms," in IEEE Access, vol. 8, pp. 30905-30917, 2020, doi: 10.1109/ACCESS.2020.2972731.
