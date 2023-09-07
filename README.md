# Empirical quantification of predictive uncertainty due to model discrepancy by training with an ensemble of experimental designs: an application to ion channel kinetics
This repository contains the code necessary to reproduce the figures presented in our publication: _Empirical quantification of predictive uncertainty due to model discrepancy by training with an ensemble of experimental designs: an application to ion channel kinetics_, Joseph G. Shuttleworth, Chon Lok Lei, Dominic G. Whittaker, Simon P. Preston and Gary R. Mirams.

This repository consists of package for the processing and analysis of patch-clamp electrophysiology data. Some of this functionality is used in the paper. The code ran to produce our figures are the `scripts` directory, and the corresponding output is provided in the `output` directory.

## Installation

These scripts has been tested with Python version 3.9 (see Dockerfile). It is recommended to install libraries and run scripts in a virtual environment to avoid version conflicts between different projects. To do this:
- Clone this repository `git clone https://github.com/CardiacModelling/empricial_quantification_of_model_discrepancy`
- Create a python virtual environment `python -m venv .venv` or (`python -m virtualenv folder_name`). If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- Activate the virtual environment using `source folder_name/bin/activate`. Simply type `deactivate` to exit the virtual environment at the end of a session.
- Install [graphviz](https://graphviz.org/). On Ubuntu, this is done by running `sudo apt install graphviz graphviz-dev`. 
- Install gcc and build essential: `sudo apt-get install gcc build-essential`
- Install cmake: `sudo apt-get install cmake`
- Install LaTeX (for plots): `sudo apt install texlive texlive-latex-extra texlive-fonts-recommended cm-super dvipng`
- Install scikit-build: `pip install scikit-build`
- Install the MarkovModels package by running `pip install -e .`.

Alternatively, you can create a [docker](https://docker.com) image using  `Dockerfile`.

## Scripts
Figure 1 was produced using `scripts/fix_wrong_param_study/simple_example.py`.

For Case I, the computations are performed using `scripts/fix_wrong_param_study/fix_wrong_params`. Then, Figure 4 is produced using `scripts/fix_wrong_param_study/big_multi` with these results.

The synthetic dataset used for Case II was produced using `scripts/fix_param_study/generate_synthetic_data.py` and this dataset was used to fit both the Wang and Beattie models using `scripts/fix_wrong_param_study/fit_all_wells_and_protocols.py`. These results are summarised using `scripts/fix_wrong_parma_study/CaseII_figure.py` and `error_compare_plot.py`.

The scripts used to produce each figure are shown in the following table:

| Figure   | script            |
| -------  | -------           |
| Fig1.pdf | simple_example.py |
| Fig3.pdf | plot_protocols.py |
| Fig4.pdf | CaseI_prediction_plots.py |
| Fig5.pdf | CaseI_main.py |
| Fig6.pdf | CaseII_prediction_plots.py |
| Fig7.pdf | CaseII_figure.py |
| Fig8.pdf | error_compare_plot.py |
| Fig10.pdf | CaseI_main.py |

## Running
To run a script execute it using Python. For example,
```python3  scripts/fix_wrong_param_study/fix_wrong_parameter.py --protocols sis staircase```

## Protocols
A list of voltage-clamp protocols are provided in  `/MarkovModels/protocols`. These are `.csv` files which describe time-series data. The filenames which correspond to the protocols used in the data are shown in the table below.

| protocol      | filename        |
| -----------   | -----------     |
| d0            | longap          |
| d1            | hhbrute3gstep'  |
| d2            | sis             |
| d3            | spacefill19     |
| d4            | staircaseramp1  |
| d5            | wangbrute3gstep |

## Results
All of the computational results mentioned in the paper are provided in the `paper_output` directory. In `paper_output`, each subdirectory includes an `info.txt` file which lists the command run to produce the output.
