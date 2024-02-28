

## Remaining work

Incorporate performance metrics into training and validation steps



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

