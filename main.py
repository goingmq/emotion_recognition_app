import argparse
from src.train import train
from src.experiment_lstm import run_lstm_experiment
from src.experiment_compare import run_comparison_experiments
from src.visualize_mfcc import generate_mfcc_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true", help="Run LSTM experiment")
    parser.add_argument("--compare", action="store_true", help="Run CNN/LSTM/CNN-LSTM comparison")
    parser.add_argument("--mfcc", action="store_true", help="Generate MFCC visualization")
    args = parser.parse_args()

    if args.compare:
        run_comparison_experiments()
    elif args.lstm:
        run_lstm_experiment()
    elif args.mfcc:
        generate_mfcc_visualization()
    else:
        train()


if __name__ == "__main__":
    main()
