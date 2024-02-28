


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

## DataExploration Notebook
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