import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from modifiedGensim.models.coherencemodel import CoherenceModel_ttc


def load_config(config_path: Path):
    """
    Load and validate processing settings from a JSON config file.
    Expected keys:
      - corpus_csv: str (path to reference corpus CSV)
      - topics_npy: str (path to .npy topics file)
      - topn: int (top-k words per topic)
      - coh_types: list[str] (coherence types, e.g., ['c_npmi'])
      - output_dir: str (directory to save CSV results)
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as cf:
        config = json.load(cf)
    required = ['corpus_csv', 'topics_npy', 'topn', 'coherence_types', 'output_dir']
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing config keys: {', '.join(missing)}")
    return config


def import_texts(corpus_path: Path):
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Reference corpus not found: {corpus_path}")
    df = pd.read_csv(corpus_path)
    if 'text' not in df.columns:
        raise KeyError(f"Expected 'text' column in {corpus_path}")
    return [str(doc).lower().split() for doc in df['text'].tolist()]


def import_topics(npy_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npy_path.is_file():
        raise FileNotFoundError(f"Topics file not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True)
    K, T, _ = data.shape
    groups = []
    for k in range(K):
        time_pairs = []
        for t in range(T - 1):
            time_pairs.append([data[k, t].tolist(), data[k, t+1].tolist()])
        groups.append(time_pairs)
    return data, np.array(groups, dtype=object)


# Computing Temporal Topic Coherence
def compute_coherence_ttc(topics: List[List[str]], texts: List[List[str]], dictionary: Dictionary, topn: int, coherence_type: str):
    cm = CoherenceModel_ttc(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
        topn=topn
    )
    return cm.get_coherence_per_topic()


# Computing Topic Coherence
def compute_coherence(topics: List[List[str]], texts: List[List[str]], dictionary: Dictionary, topn: int, coherence_type: str):
    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
        topn=topn
    )
    return cm.get_coherence_per_topic()


def topic_smoothness(topics: List[List[str]], topn: int):
    K = len(topics)
    scores = []
    for i, base in enumerate(topics):
        base_set = set(base[:topn])
        overlaps = [len(base_set & set(other[:topn])) / topn for j, other in enumerate(topics) if j != i]
        scores.append(sum(overlaps) / len(overlaps))
    return float(sum(scores) / K)


def compute_ttq(total_topics: np.ndarray, group_topics: np.ndarray, texts: List[List[str]], dictionary: Dictionary, topn: int, coherence_type: str):
    print("Computing TTQ")
    K, T, _ = total_topics.shape
    all_coh_scores = []
    avg_coh_scores = []
    for k in range(K):
        print(k)
        avg_coh_scores.append(float(np.mean(compute_coherence_ttc(total_topics[k].tolist(), texts, dictionary, topn, coherence_type))))
        all_coh_scores.append(compute_coherence_ttc(total_topics[k].tolist(), texts, dictionary, topn, coherence_type))
    avg_smooth_scores = []
    all_smooth_scores = []
    for k in range(K):
        print(k)
        pair_scores = [topic_smoothness(pair, topn) for pair in group_topics[k]]
        avg_smooth_scores.append(float(np.mean(pair_scores)))
        all_smooth_scores.append(pair_scores)
    return pd.DataFrame({
        'topic_idx': list(range(K)),
        'temporal_coherence': all_coh_scores,
        'temporal_smoothness': all_smooth_scores,
        'avg_temporal_coherence': avg_coh_scores,
        'avg_temporal_smoothness': avg_smooth_scores
    })


def compute_yearly_tq(yearly_topics: np.ndarray, texts: List[List[str]], dictionary: Dictionary, topn: int, coherence_type: str):
    print("Computing Yearly TQ")
    T = yearly_topics.shape[0]
    all_coh, avg_coh, div = [], [], []
    for t in range(T):
        all_coh.append(compute_coherence(yearly_topics[t].tolist(), texts, dictionary, topn, coherence_type)
        )
        avg_coh.append(
            float(np.mean(compute_coherence(yearly_topics[t].tolist(), texts, dictionary, topn, coherence_type)))
        )

        div.append(1-topic_smoothness(yearly_topics[t].tolist(), topn))
    return pd.DataFrame({
        'year': list(range(T)),
        'all_coherence': all_coh,
        'avg_coherence': avg_coh,
        'diversity': div
    })


def main():
    # Load config
    import argparse
    parser = argparse.ArgumentParser(description="Compute DTQ metrics via JSON config.")
    parser.add_argument('config', type=Path, help="Path to JSON config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    corpus_csv = Path(config['corpus_csv'])
    topics_npy = Path(config['topics_npy'])
    topn = int(config['topn'])
    coherence_types = config['coherence_types']
    output_dir = Path(config['output_dir'])

    texts = import_texts(corpus_csv)
    dictionary = Dictionary(texts)
    total, groups = import_topics(topics_npy)
    yearly = total.transpose(1, 0, 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    for coh in coherence_types:
        ttq_df = compute_ttq(total, groups, texts, dictionary, topn, coh)
        ttq_df['ttq_product'] = ttq_df['avg_temporal_coherence'] * ttq_df['avg_temporal_smoothness']
        ttq_average = ttq_df['ttq_product'].mean()
        print("TTC, TTS, TTQ :",ttq_df['avg_temporal_coherence'].mean(),ttq_df['avg_temporal_smoothness'].mean(),ttq_average)

        tq_df = compute_yearly_tq(yearly, texts, dictionary, topn, coh)
        tq_df['tq_product'] = tq_df['avg_coherence'] * tq_df['diversity']
        tq_average = tq_df['tq_product'].mean()
        print("TC, TD, TQ :", tq_df['avg_coherence'].mean(), tq_df['diversity'].mean(),tq_average)
        ttq_df.to_csv(str(output_dir) + f"\\ttq_{coh}.csv", index=False)
        tq_df.to_csv(str(output_dir) + f"\\tq_{coh}.csv", index=False)


if __name__ == '__main__':
    main()
