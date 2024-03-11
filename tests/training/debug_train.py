from pathlib import Path

from cats_and_dogs_hai.training.run_train import run_classification_train
from cats_and_dogs_hai.training.run_train import run_segmentation_train

def classification_main():
    parent_dir = Path("debug_runs")
    sdir=  parent_dir / 'classification'
    sdir.mkdir(exist_ok=True, parents=True)
    run_classification_train(max_epochs=2, save_dir=sdir, debug=True, accelerator="cpu")

def segmentation_main():
    parent_dir = Path("debug_runs")
    sdir=  parent_dir / 'segmentation'
    sdir.mkdir(exist_ok=True, parents=True)
    run_segmentation_train(max_epochs=2, save_dir=sdir, debug=True, accelerator="cpu")


if __name__ == "__main__":
    print('------\n')
    print('Running Classification')
    classification_main()

    print('\n------\n')
    print('Running Segmentation')
    segmentation_main()