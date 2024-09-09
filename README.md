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

## Final results
```
| Model                           |   cola |    rte | mrpc           | stsb           |   sst2 |   qnli |   mnli | qqp            |   Average Score |
|---------------------------------|--------|--------|----------------|----------------|--------|--------|--------|----------------|-----------------|
| bert-base-uncased (baselines)   | 0.5735 | 0.6462 | 0.8382, 0.8862 | 0.8838, 0.8799 | 0.9232 | 0.914  | 0.8415 | 0.9117, 0.8807 |          0.8344 |
| bert-base-uncased (agni_5)      | 0.5677 | 0.639  | 0.8456, 0.8919 | 0.8850, 0.8817 | 0.9266 | 0.9143 | 0.8429 | 0.9113, 0.8802 |          0.8351 |
| albert-base-v1 (baselines)      | 0.4875 | 0.7148 | 0.8505, 0.8924 | 0.8887, 0.8858 | 0.8922 | 0.9065 | 0.8222 | 0.8990, 0.8639 |          0.8276 |
| albert-base-v1 (agni_5)         | 0.4894 | 0.704  | 0.8554, 0.8952 | 0.8900, 0.8862 | 0.8888 | 0.9057 | 0.8212 | 0.8992, 0.8641 |          0.8272 |
| t5-base (baselines)             | 0.4617 | 0.5451 | 0.7083, 0.8205 | 0.7532, 0.7760 | 0.9335 | 0.9138 | 0.864  | 0.9061, 0.8746 |          0.7779 |
| t5-base (agni_5)                | 0.4751 | 0.5487 | 0.7059, 0.8187 | 0.7625, 0.7757 | 0.9358 | 0.9143 | 0.8642 | 0.9062, 0.8746 |          0.7802 |
| squeezebert-uncased (baselines) | 0.3958 | 0.6462 | 0.7990, 0.8620 | 0.8688, 0.8719 | 0.8922 | 0.8991 | 0.81   | 0.8960, 0.8636 |          0.8004 |
| squeezebert-uncased (agni_5)    | 0.4018 | 0.6462 | 0.8064, 0.8654 | 0.8686, 0.8707 | 0.8991 | 0.9001 | 0.8104 | 0.8953, 0.8617 |          0.8023 |
| bart-base (baselines)           | 0.426  | 0.6931 | 0.8456, 0.8923 | 0.8758, 0.8759 | 0.9323 | 0.9193 | 0.8611 | 0.9054, 0.8726 |          0.8272 |
| bart-base (agni_5)              | 0.4207 | 0.6931 | 0.8480, 0.8946 | 0.8784, 0.8788 | 0.9346 | 0.9206 | 0.8603 | 0.9047, 0.8722 |          0.8278 |
```
