import logging
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
T0_DATA_PATH = DATA_PATH / "raw_data" / "t0"
T12_DATA_PATH = DATA_PATH / "raw_data" / "t12-updated"
PLOT_PATH = ROOT_PATH / "plots"
RANDOM_STATE = 42
LOGGER_LEVEL = logging.INFO
DATASET_NAME = "combined_t0_t12_datasets_updated.csv"
LABEL_DICT = {0: "HC", 1: "Stage_0", 2: "Stage_1a", 3: "Stage_1b", 4: "Transition"}
