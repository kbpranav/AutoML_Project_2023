# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project

This repository contains all things needed for the final projects.
Your task is to optimize a NN by AutoML means.
For details, please refer to the project PDF.

### (Recommended) Setup new clean environment

Use a package manager, such as the one provided by your editor, python's built in `venv`
or [miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements).

#### Conda
Subsequently, *for example*, run these commands, following the prompted runtime instructions:
```bash
conda create -n automl python=3.10
conda activate automl
pip install -r requirements.txt
```

#### Venv

```bash
# Make sure you have python 3.8/3.9/3.10
python -V
python -m venv my-virtual-env
./my-virtual-env/bin/activate
pip install -r requirements.txt
```

#### SMAC
If you have issues installing SMAC,
follow the instructions [here](https://automl.github.io/SMAC3/main/1_installation.html).


### Data
You need to pre-download all the data required by running `python datasets.py`.

Stores by default in a `./data` directory. Takes under 20 seconds to download and extract.

### Tips

All code we provide does consider validation and training sets.
You will have to implement a method to use the test set yourself.

#### `multi_fidelity_template.py`
* Example of how to use SMAC with multi-fidelity optimization.
* The example uses image size as the fidelity.
* To get quick results, you can lower the image size to 4x4 for a quick debug signal if you like.
However, make sure when comparing to any baseline to always use the maximum fidelity of 32x32.
* The configsapce that we used to get the baseline performance is in default_configspace.json.


### mf_default.py
* This script leverages SMAC for multifidelity optimization. It operates within a basic default configuration space.

### mf_proxy_final.py
* Utilizing the new configuration space and integrating proxy optimization with warm-start techniques, this code aims to enhance model performance.

### mf_proxy_evolution.py
* By incorporating Regularized Evolution alongside Proxy optimization, this script aims to further refine final model performance.

### earlystopping.py
* This Python file encapsulates the implementation of the early stopping mechanism, a crucial component for improving optimization efficiency.

