
Task based on modified cats vs dogs challenge from: https://github.com/harrison-ai/hai-tech-tasks/blob/develop/cats_and_dogs.md

To establish a baseline a ResNet18 model is used for breed classification with the outputs of this model used to differentiate between cat and dog. 


## Remaining work

* Add segmentation model training, performance evaluation and inference code.
  * Was going to use [lraspp_mobilenet_v3_large](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights) as a starting point. 
* Modify inference code to work with segmentation, likely to create a master inference module which runs both models.



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

