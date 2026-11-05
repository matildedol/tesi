# A Modern Look at Classical Generalization Bounds

This repository contains the source code and LaTeX files used for producing my master's thesis *A Modern Look at Classical Generalization Bounds*. You can find it at ```tex/tesi.pdf```. 

## Contents

The folder ```tex``` contains the .tex files and the thesis .pdf, as well as the figures appearing in the thesis, inside ```/outputs```. 

The folder ```source```contains the code used for running experiments.
- ```experiments.py``` is the main Python code that produces all the plots in the main text. Call it from your CLI with the argument --curves followed by the curves you desire to plot among gap, test_err, train_err, bound_union, bound_radmass, bound_noise. All the hyperparameters can be tuned with their specific arguments from the CLI. Moreover, the arguments --out followed by a path and --show allow to save and show the resulting figure, respectively. More details below. 
- ```make_figures.sh``` allows to produce *all* the figures in the main tex in a single run.  

- ```notebooks/experiments.ipynb``` is the Jupiter Notebook version of the file above. It contains the same experiments with descriptions and references to the main text. It relies on ```notebooks/base.ipynb``` for the definition of the main functions used. 

- ```notebooks/prel_experiments.ipynb``` is the Jupiter Notebook for producing all the figures in the Appendix. 

Lastly, the folder ```extra``` contains additional files, both tex ones and code, with my notes, reasonings, and other experiments I did while working at this thesis to better understand the math or collect ideas. Rather messy folder.

## How to use the code

```requirements.txt``` contains all the requirements for running both Python and Jupiter Notebook files. 

To run ```experiments.py```, remember to change paths as needed. Right now they should be ../tex/outputs. 

Use ```./make_figures.sh``` to produce all figures in one shot. 


Usage of ```experiments.py```:   

```experiments.py [-h]
[--curves {gap,test_err,train_err,bound_union,bound_radmass,bound_noise,train_bound_union,train_bound_radmass,train_bound_noise} [{gap,test_err,train_err,bound_union,bound_radmass,bound_noise,train_bound_union,train_bound_radmass,train_bound_noise} ...]]
[--train_size TRAIN_SIZE] [--test_size TEST_SIZE] [--supp_size SUPP_SIZE] [--num_iter NUM_ITER] [--noise_type {gaussian,bernoulli}] [--noise_level NOISE_LEVEL [NOISE_LEVEL ...]] [--num_crossing NUM_CROSSING] [--C_union C_UNION] [--C_radmass C_RADMASS] [--C_noise C_NOISE] [--seed SEED] [--n_jobs N_JOBS] [--out OUT] [--show]

