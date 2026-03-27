import os
import numpy as np
import librosa
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

K_VALUES = [32, 48, 64, 96]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def extract_mfcc_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=NUM_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Mean-pool across all time frames so the full audio clip is used
    return mfccs.mean(axis=1)  # shape: (NUM_MFCC,)


def load_dataset():
    X = []
    y = []

    for root, _, files in os.walk(DATASET_PATH):
        for file_name in files:
            if not file_name.endswith((".wav", ".mp3")):
                continue

            file_path = os.path.join(root, file_name)
            label = os.path.basename(root)
            try:
                X.append(extract_mfcc_features(file_path))
                y.append(label)
            except Exception as exc:
                print(f"Skipping {file_path}: {exc}")

    return np.array(X), np.array(y)


def build_cluster_label_map(cluster_ids, labels, n_clusters):
    cluster_to_label = {}
    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        if np.any(mask):
            most_common = Counter(labels[mask]).most_common(1)
            if most_common:
                cluster_to_label[cluster_id] = most_common[0][0]
    return cluster_to_label


def evaluate_k(X_train, X_test, y_train, y_test, n_clusters):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X_train_scaled)

    train_cluster_ids = kmeans.predict(X_train_scaled)
    cluster_to_label = build_cluster_label_map(train_cluster_ids, y_train, n_clusters)

    test_cluster_ids = kmeans.predict(X_test_scaled)
    y_pred = [cluster_to_label.get(cluster_id, "Unknown") for cluster_id in test_cluster_ids]

    labels = sorted(list(set(y_test.tolist()) | set(y_pred)))
    recalls = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recall_map = {label: recall for label, recall in zip(labels, recalls)}

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "k": n_clusters,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "chaos_recall": recall_map.get("Chaos", 0.0),
        "crowd_recall": recall_map.get("Crowd", 0.0),
    }


def main():
    print("=" * 70)
    print("K Sweep for ZonoTrack")
    print("=" * 70)
    print(f"Testing K values: {K_VALUES}")

    X, y = load_dataset()
    if len(X) == 0:
        print("No audio files found in dataset.")
        return

    print(f"Loaded {len(X)} samples across {len(set(y.tolist()))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    for k in K_VALUES:
        print(f"\nEvaluating K={k}...")
        metrics = evaluate_k(X_train, X_test, y_train, y_test, k)
        results.append(metrics)
        print(
            f"K={k} | Acc={metrics['accuracy']:.4f} | "
            f"MacroF1={metrics['macro_f1']:.4f} | "
            f"WeightedF1={metrics['weighted_f1']:.4f} | "
            f"ChaosRecall={metrics['chaos_recall']:.4f} | "
            f"CrowdRecall={metrics['crowd_recall']:.4f}"
        )

    results_sorted = sorted(
        results,
        key=lambda r: (r["macro_f1"], r["chaos_recall"] + r["crowd_recall"], r["accuracy"]),
        reverse=True,
    )

    best = results_sorted[0]

    print("\n" + "-" * 70)
    print("Summary")
    print("-" * 70)
    for row in results:
        print(
            f"K={row['k']:<3} Acc={row['accuracy']:.4f} "
            f"MacroF1={row['macro_f1']:.4f} WeightedF1={row['weighted_f1']:.4f} "
            f"ChaosRecall={row['chaos_recall']:.4f} CrowdRecall={row['crowd_recall']:.4f}"
        )

    print("\nBest K by MacroF1 + minority recall tie-break:")
    print(
        f"K={best['k']} | Acc={best['accuracy']:.4f} | MacroF1={best['macro_f1']:.4f} | "
        f"ChaosRecall={best['chaos_recall']:.4f} | CrowdRecall={best['crowd_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
