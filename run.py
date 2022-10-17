from prepare_finetuning_data import prepare_data
from train_sbert import run_trainsbert
import argparse
from get_representations import get_representations_datasets
from compute_distance_matrix import *
from utils_sim import correlation_matrices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--finetuning_year", type=str, default="2017-2013")
    parser.add_argument("--analysis_year", type=str, default="2021")
    args = parser.parse_args()

    if args.train:
        prepare_data(args.finetuning_year)
        print("Start training SBERT - may take a while...")
        run_trainsbert(args.finetuning_year)

    get_representations_datasets(args.finetuning_year, args.train)
    compute_distances()
    compute_distances(do_whiten=True)
    correlation_matrices(args.analysis_year)
