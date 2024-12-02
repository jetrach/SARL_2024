This repository contains the task code, data, and analysis code for Trach et al. "Rewards transiently and automatically enhance sustained attention." 

Contents:
task - this folder has the task code for each experiment. Stimuli are stored in imgs folder. 

analysis - 
  This folder has a folder for each experiment that contains:
      - data - the subject-wise data
      - SARL_ExptX_Analysis.Rmd - the main R Markdown file that executes the main analyses
      - SARL_ExptX_modelingInfo.csv - csv file of the subject-wise fitted parameters and AIC/BIC values. This is used in the markdown script and added to the compiled data
      - SARL_ExptX_rawdata_nXX_final.csv - All the subject-wise data compiled into one spreadsheet to facilitate running the analyses (the compiling process can take quite a while for Experiments 2 and 3.
      - SARL_ExptX_RPE_1a - simulated trialwise RPEs generated with the single alpha model fitted parameters for each subject
      - SARL_ExptX_RPE_2a - simulated trialwise RPEs generated with the two alpha model fitted parameters for each subject
  Additionally, this folder has a modeling folder that contains the modeling code for Experiments 1 and 2:
      - func_1alpha.m - single learning rate (alpha) model
      - func_2alpha.m - two learning rate (alpha) model
      - mcdougle_softmax_func.m - softmax choice function for fitting
      - SARL_modeling.m - fitting script for the model, this uses the compileds (and cleaned) data in the csv file to fit the RL models
      - simModel.m - this script uses the fitted model parameters to simulate behavior based on these parameters
      - models - this folder contains a .mat file for each of the two RL models that we used in the manuscript.
