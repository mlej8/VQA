# VQA

## Setup
In order to setup the datasets locally, you would need around 35 GB of free disk space.

Simply run the `fetchdata.sh` script and it will setup the data directory for you. 

The official instructions to download the datasets and preprocessing tasks for the VQA task can be found here: https://github.com/GT-Vision-Lab/VQA/blob/master/README.md

# Conda environment
Add conda-forge to channels: `conda config --add channels conda-forge`

To create an environment using requirements.txt use: `conda create --name vqa --file requirements.txt`

Alternatively create a python virtual environment with: `python -m venv vqa`

Activate virtual environment with: `conda activate vqa` or `source vqa/bin/activate`

Install requirements if using python virtual environment: `pip install -r requirements.txt`

To update requirements file with conda: `conda list -e > requirements.txt`

To update requirements file with pip: `pip freeze > requirements.txt`

# Download VQA datasets
Run the bash script `./fetchdata.sh` to download all datasets required for VQA v2.0 real images task. This script also sets up the directory structure.

datasets/
    - Annotations
    - Complementary Pairs
    - Images
    - Questions

# Preprocessing

In order to build the vocabulary for the questions and the answers, run `python -m preprocessing.preprocess_text`. This will create a directory `datasets/Vocabulary` where the vocabularies for questions and answers are stored.

# Models

To run the simple VQA baseline: `python simple_vqa_baseline.py`
## Credits
The PythonHelperTools, PythonEvaluationTools, QuestionTypes and Results comes from the following repository: https://github.com/GT-Vision-Lab/VQA


