import argparse
import logging
import os

from .data_loader import load_and_prepare_data
from .train import run_training, run_testing

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'e_commerce_audio')
METADATA_CSV = os.path.join(AUDIO_DIR, 'metadata.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch():
    logging.info("=== [FETCH MODE] ===")
    df = load_and_prepare_data()
    if df is not None and not df.empty:
        logging.info(f"Fetched and saved {len(df)} audio files and metadata.")
    else:
        logging.warning("No data fetched.")

def train():
    logging.info("=== [TRAIN MODE] ===")
    if not os.path.exists(METADATA_CSV):
        logging.error(f"Metadata CSV not found. Run fetch mode first. {METADATA_CSV}")
        return
    run_training(metadata_csv=METADATA_CSV)

def test():
    logging.info("=== [TEST MODE] ===")
    if not os.path.exists(METADATA_CSV):
        logging.error("Metadata CSV not found. Run fetch mode first.")
        return
    run_testing(metadata_csv=METADATA_CSV)

def main():
    parser = argparse.ArgumentParser(description="E-commerce Command Recognition Pipeline")
    parser.add_argument('--mode', choices=['fetch', 'train', 'test'], required=True, help="Pipeline mode to run")
    args = parser.parse_args()

    if args.mode == 'fetch':
        fetch()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()

if __name__ == '__main__':
    main()
