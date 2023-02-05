# Data, scripts and notebooks to analyse the simulations from:
ref


Content:
- ```setup_md/``` contains mdp files, force field files, examplary setup scripts and everything else need to set up the MD simulations for each system.
- ```specdens_mapping/``` contains example scripts to calculate NMR relaxation rates using the spectral density mapping approach (https://pubs.acs.org/doi/10.1021/acs.jctc.0c01338, https://doi.org/10.1039/C8CP03915A).
- ```reweighting/``` contains scripts to perform k-fold cross validation to determine an adequate $\theta$ for reweighting with ABSURDer.
- ```notebooks/``` contains Jupyter notebooks to recreate all the plots in the main and SI of the manuscript and a current version of ABSURDer.
- ```data/``` contains all the data relevant to recreate the plots.
- ```modify_topology/``` contains a pyhton script that can be used to modify a GROMACS topology file implementing our force field changes. Only tested on the AMBER force field a99SB*-ILDN. Usage: ```python moddihed_ffAMBER.py topology.top```

Trajectories and other simulation files are available at:<br>
Download link will be available soon.
