from pathlib import Path

from cats_and_dogs_hai.training.create_trainer import run_train

def main():
    sdir= Path("debug_runs")
    run_train(max_epochs=2, save_dir=sdir, debug=True, accelerator="cpu")

if __name__ == "__main__":
    main()