import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OGB_ROOT = os.path.join(REPO_ROOT, "dataset")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DEFAULT_MAIN_T = 10000
DEFAULT_RUNS = 10
DEFAULT_WORKERS = 10
DEFAULT_N_NEG = 9
DEFAULT_HIDDEN = 100
DEFAULT_EE_NET_POOL_STEP = 50

TRAIN_EVERY_BEFORE_2000 = 50
TRAIN_EVERY_AFTER_2000 = 100

MAIN_DATASET_CONFIGS = {
    "MovieLens": {
        "path": os.path.join(DATA_DIR, "MovieLens/movie_2000users_10000items_noedge.npy"),
        "n_users": 2000,
        "n_items": 10000,
    },
    "Amazon_fashion": {
        "path": os.path.join(DATA_DIR, "Amazon_fashion/new/Insert/Amazon_fashion_4000users_noedge.npy"),
        "n_users": 4000,
        "n_items": 4000,
    },
    "Facebook": {
        "path": os.path.join(DATA_DIR, "Facebook/Insert/facebook_combined_ALLusers_noedge.npy"),
        "n_users": 4039,
        "n_items": 0,
    },
    "Grqc": {
        "path": os.path.join(DATA_DIR, "GrQc/Insert/GrQc_ALLusers_noedge.npy"),
        "n_users": 5242,
        "n_items": 0,
    },
    "PPA": {
        "path": OGB_ROOT,
        "n_users": 0,
        "n_items": 0,
    },
    "Collab": {
        "path": OGB_ROOT,
        "n_users": 0,
        "n_items": 0,
    },
    "Vessel": {
        "path": OGB_ROOT,
        "n_users": 0,
        "n_items": 0,
    },
}

DEFAULT_BASELINE_DATASETS = ["MovieLens", "Amazon", "Facebook", "GrQc", "Collab", "PPA", "Vessel"]
DEFAULT_BASELINE_METHODS = ["EE-Net", "NeuralUCB", "NeuralTS", "NeuralGreedy", "LinUCB", "KernelUCB"]
