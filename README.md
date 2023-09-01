## Installation and Getting Started
1. Request raw data from authors and place in `./data/raw_data/t12-updated/`
2. Install project with one of the two following methods:
    1. Using `conda`, install environemnt with `conda env create --file environment.yml`, or 
    2. Using `vscode` and `docker`, open this folder in vscode and run command `> Dev Containers: Reopen in container`
3. Copy `.env.template` into `.env` and add your weights and biases API key and path to this directory
4. Review the settings in `./configs/config.yaml`. As currently set, this config will create a model that matches our results reported in the paper. 
5. Run training script with `./procan_connectome/main.py` from this directory. Config settings can be overridden using command line arguments. I.e., `python ./procan_connectome/main.py  pipeline.rfecv=False` will run the training script **without** `rfecv` feature selection. 

## Generating figures
1. Download run results from `wandb` by running command `python ./procan_connectome/utils/download_wandb_run_table.py`. You will need access to the wandb project to obtain our results. Please contact the authors if you'd like the raw results .csv files. 
2. Run notebook `./eda/updated_results.ipynb`
3. Figures will be output to `.plots`

## Existing model checkpoints
You can review our trained models by unpickling the files found in `./data/trained_models`. Note that these are pickled `LOOCV_Wrapper` instances. See `./procan_connectome/model_training/loocv_wrapper.py` for more details on this classes attributes and member functions. 