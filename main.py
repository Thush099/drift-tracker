
import argparse
import logging
from src.pipeline import Pipeline
from src.utils import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description='Drift Tracker')
    parser.add_argument('--data-size', type=int, default=1000, help='Size of the synthetic data')
    parser.add_argument('--drift-threshold', type=float, default=0.1, help='Threshold for drift detection')
    args = parser.parse_args()

    # Generate synthetic data
    data = generate_synthetic_data(args.data_size)

    # Create pipeline
    pipeline = Pipeline(data, args.drift_threshold)

    # Run pipeline
    pipeline.run()

if __name__ == '__main__':
    main()
