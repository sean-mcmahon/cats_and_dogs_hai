
Task based on modified cats vs dogs challenge from: https://github.com/harrison-ai/hai-tech-tasks/blob/develop/cats_and_dogs.md

To establish a baseline a ResNet18 model is used for breed classification with the outputs of this model used to differentiate between cat and dog. 
For the mask prediction a version of EfficientNet is used.

Classification and segmentation are performed and trained separately to establish a baseline. The combined inference is one call though so in future if these two models are combined the inference calls made externally do not change.


## Remaining work

* Implement inference with REST API to integrate your predictive model into a backend system.
* Plan how to scale up to 8000 cases per second.


## Installation and Setup

* Install [Python](https://www.python.org/)
* Install Conda, either [full version](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mini-conda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
* Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```
conda env create -f environment.yaml   
conda activate cats_and_dogs_hai
poetry install
```

To set up kernel for Jupyter-Lab 
```
python -m ipykernel install --user --name cats_and_dogs_hai
```

## Dataset

Download cat_and_dogs.zip into repository root
```
cd <repository root>
mkdir -p data/cats_and_dogs && cd data/cats_and_dogs
curl -LO https://github.com/harrison-ai/hai-tech-tasks/releases/download/v0.1/cats_and_dogs.zip .
unzip cats_and_dogs.zip -d .
```

## Data Exploration Notebook
```
conda activate cats_and_dogs_hai
jupyter-lab notebooks/DataExplore.ipynb
```

## Tests

``` 
conda activate cats_and_dogs_hai
python -m pytest tests/
```

To run a simple training loop (excluded pytest to get full stdout and reduce test time)
```
python tests/training/debug_train.py
```

