# A Modern Look at Classical Generalization Bounds

This repository contains the source code and LaTeX files used for producing my master's thesis. 

The folder ```tex``` contains the .tex files and the thesis .pdf, as well as the figures appearing in the thesis inside ```/outputs```. 

The folder ```source```contains the Python code used for running experiments. The code is simple and 100% Jupiter Notebook. The folder contains:
- ```base.ipynb``` that is the setup code where we define Python functions for running the experiments (the central ones are ```main_loop()``` that fits models and estimates the generalization gap and ```plots()``` that produces the usual 3 plots appearing in the thesis) and the functions of the bounds;
- ```experiments.ipynb``` that is the code where experiments are actually run, producing all the figures in Sections 2 to 5;
- ```prel_experiments.ipynb``` that produces all the figures in Appendix A. 



