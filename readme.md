# Codes for "The finite state projection based Fisher information matrix approach to estimate information and optimize single-cell experiments" 

This repository contains the codes necessary to reproduce figures from the above manuscript. All codes are implemented in python 2.7. 


* The codes are organized by the _model_ being used (i.e. birth-death, bursting gene, or toggle models). 
* Each model has a code which define the model structure, for example `birth_death.py`.
* Analysis scripts for the different models are the other python files in each folder. 
* Note that many of these analyses, especially for the toggle and bursting gene model, rely upon expensive computations that were performed on the [W.M. Keck Compute Cluster](https://www.engr.colostate.edu/ens/info/researchcomputing/cluster/). For convenience, only the results of these numerical simulations are included in this repository. 

