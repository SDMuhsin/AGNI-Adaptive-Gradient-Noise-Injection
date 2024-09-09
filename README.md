# Adaptive Gradient Noise Injection

This repository houses the source code for the paper titled AGNI : Adaptive Gradient Noise Injection for Improved Transformer Fine Tuning. This work seeks to improve fine tuning performance of neural network training by introducing guided noise into the gradients during the backward pass.

## Environment and install instructions

The script `install.sh` contains the commands to install required python dependencies. Note that this script is tailored for slurm clusters. Some libraries may need to be manually installed from source.
Running the code additionally required GNU parallel, but this can be avoided by modifying the scripts.

## Project structure

- Source code
	- All source code for running experiments, including vizualizations exist in the `/source` directory
	- `/source/run_glue_agni2.py`holds code to run models with AGNI + AdamW on GLUE datasets
	- `/source/run_glue_baselines.py` holds code to generate baselines with just AdamW for different models on GLUE datasets
	- `/source/run_glue_few_other_baselines.py` is largely the same as above, but allows switching of optimizers from AdamW. This is used to identify best baseline optimizer
	- `/source/consolidate.py` is used to tabulate the result of these runs based on job\_id
	- `/source/viz_runtimes.py` visualizes runtimes using data saved in `/saves` during the running of above scripts
- Running scripts
	- `/scripts` houses scripts to execute all experiments with different input arguments
	- `/scripts/run_glue_agni.sh` runs different models on GLUE with AGNI+AdamW.
	- `/scripts/run_glue_baselines.sh` runs the baselines with just AdamW
	- `/scripts/run_glue_few_other_baselines.sh` is largely the same as above but allows to switch between optimizers
	- `/scripts/cluster*` scripts are the same as above, but is written for running on SLURM GPU clusters with GNU parallel

## Steps to reproduce experiments

1. Install environments using `install.sh`
2. Run baselines using `/scripts/run_glue_baselines.sh` for required models
3. Run AGNI+AdamW case using `/scripts/run_glue_agni.sh` for required models. These two steps save results in `/saves`
4. Specify the same models, tasks and job\_ids used in above two steps in `source/consolidate.py` to tabulate results

