import json
import argparse
from pathlib import Path
import numpy as np


def load_config(config_path: Path):
    """
    Load and validate processing settings from a JSON config file.
    Expected keys:
      - input_dir: str (path to directory containing .txt topic files)
      - n_topics: int (number of topics per time slice)
      - n_words: int (number of top words per topic)
      - output_dir: str (directory to save the resulting NumPy file)
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as cf:
        config = json.load(cf)

    required_keys = ['input_dir', 'n_topics', 'n_words', 'output_dir']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing keys in config file: {', '.join(missing)}")

    return config


def load_topic_file(file_path: Path):
    """
    Read a single topic file and return a list of topics,
    each represented as a list of word tokens.
    """
    topics = []
    for line in file_path.read_text().splitlines():
        text = line.strip()
        tokens = text.split()
        topics.append(tokens)
    return topics


def get_topics(input_dir: Path, n_topics: int, n_words: int, output_dir: Path):
    """
    Load topics from text files in `input_dir`, arrange into a
    NumPy array of shape (topics, time_slices, words), and save
    as 'topics.npy' in `output_dir`.
    """
    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    total_time = len(txt_files)
    data = np.empty((total_time, n_topics, n_words), dtype=object)

    for idx, file_path in enumerate(txt_files):
        topics = load_topic_file(file_path)
        if len(topics) != n_topics:
            raise ValueError(
                f"Expected {n_topics} topics in {file_path}, got {len(topics)}"
            )
        data[idx, :, :] = topics

    # Transpose to (topics, time, words)
    output = data.transpose(1, 0, 2)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(output_dir) + '/topics.npy'
    np.save(save_path, output)
    print(f"Saved topics to {save_path}")

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process topic model outputs into a consolidated NumPy array using a JSON config file."
    )
    parser.add_argument(
        'config', type=Path,
        help="Path to JSON configuration file."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    input_dir = Path(config['input_dir'])
    n_topics = int(config['n_topics'])
    n_words = int(config['n_words'])
    output_dir = Path(config['output_dir'])

    get_topics(
        input_dir=input_dir,
        n_topics=n_topics,
        n_words=n_words,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
