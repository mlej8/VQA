# VQA

## Setup
In order to setup the datasets locally, you would need around 35 GB of free disk space.

Simply run the `fetchdata.sh` script and it will setup the data directory for you. 

The official instructions to download the datasets and preprocessing tasks for the VQA task can be found here: https://github.com/GT-Vision-Lab/VQA/blob/master/README.md

# Conda environment
Add conda-forge to channels: `conda config --add channels conda-forge`

To create an environment using requirements.txt use: `conda create --name vqa --file requirements.txt`
Activate environment: `conda activate vqa`


To update requirements file: `conda list -e > requirements.txt`

## Credits
The Python Helper and Evaluation tools comes from the following repository: https://github.com/GT-Vision-Lab/VQA


