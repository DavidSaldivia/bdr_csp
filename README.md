# BDR_MCRT
Python codes for optical analysis of Beam Down Receivers (BDR) using MonteCarlo Ray Tracing (MCRT) method. This repository contains the modules and example scripts for the work developed in my PhD thesis.
This repository originally was created to have examples of the codes used in the publication "Optical analysis and optimization of a beam-down receiver for advanced cycle concentrating solar thermal plants" (Applied Thermal Engineering, 2021). These codes are now in the folder '2_Optical_Analysis_old'. New version of the code are in '2_Optical_Analysis'. The main difference between both is that the old version used iterations over datasets and parallelisation. The new code is completely vectorised and therefore quicker.

Folders '4_SBR_Model' and '5_SPR_Models' have the different receiver models. Folder '6_Thermal_Subsys_Optim' is the thermal optimisation (minimising the LCOH), while '7_Overall_Optim_Dispatch' has the dispatch model and the performance simulations under real conditions in Australia.
The four modules developed in my thesis are:
- 'BeamDownReceiver.py' in folder '2_Optic_Analysis'. Imported as BDR.
- 'SolidBlockReceiver.py' in folder '4_SBR_Model'. Imported as SBR.
- 'SolidParticleReceiver.py' in folder '5_SPR_Models'. Imported as SPR.
- 'PerformancePowerCycle.py' in folder '7_Overall_Optim_Dispatch'. Imported as PPC.

BDR and SBR can run on their own. SPR requires BDR, and PPC requires SPR and BDR. As a recommendation, download all the folders to ensure they work properly. Also, the folder names are included when the modules are imported.

To run any code, first, you need to create a ray dataset with SolarPILOT. For this:
1. Launch SolarPILOT and 'Open' a project (Ctrl+O). Look for the file '0-SolarPilot_project.spt' in the folder '0_Data/MCRT_Datasets_Final'. This file contains the parameters used in my Thesis.
2. Go to the Layout setup and select the tower height. Go to the tab Performance Simulation, and select SolTrace in 'Flux simulation model'. Tick "Save all raytrace data" and click on the left to select where to save the file. Then click 'Simulate performance'.
3. Additionally, you can also run a script to generate several datasets. For example, go to 'open script' and select '0-script_rays_height.lk'. This allows to generate datasets between 20 and 100 m tower height (of course you can edit that too). It is important, before run the script, do step 2 and save the dataset file as "MCRT_TempDataset.csv". The script will rename that file for the specific height.

To run the dispatch model you need MERRA-2 and NEM data. in '0_Data' there are scripts to download and process these files.
