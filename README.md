# BDR_MCRT
Python codes for optical analysis of Beam Down Receivers (BDR) using MonteCarlo Ray Tracing (MCRT) method.
In this repository two main code files are presented:
  - BDR.py: It is the module with different functions to calculate the interceptions with BDR optical devices and calculate the final optical efficiencies per heliostat.
  - BDR_example.py: An example about how to run the functions contained in previous module.
 
In addition, the SolarPILOT project file to generate the dataset is presented. It should be open with SolarPILOT, generate the layout and run a performance simulation using SolTrace. This should generate a file with dataset, which is later read by the BDR_example.

Additionally, plots with the radiation map for the base case considered are presented.
