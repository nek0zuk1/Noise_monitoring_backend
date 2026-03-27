import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
N_CLUSTERS = 96
TEST_SIZE = 0.2
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')


def extract_mfcc_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=NUM_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Mean-pool across all time frames so the full audio clip is used
    return mfccs.mean(axis=1)  # shape: (NUM_MFCC,)


def load_dataset():
    X, y = [], []
    for root, _, files in os.walk(DATASET_PATH):
        for file_name in files:
            if not file_name.endswith(('.wav', '.mp3')):
                continue
            file_path = os.path.join(root, file_name)
            label = os.path.basename(root)
            try:
                X.append(extract_mfcc_features(file_path))
                y.append(label)
            except Exception as e:
                print(f"  Error: {file_name}: {e}")
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


def main():
    print("=" * 60)
    print(" ZonoTrack K-Means Confusion Matrix Visualization")
    print("=" * 60)

    print("\nLoading dataset...")
    X, y = load_dataset()
    if len(X) == 0:
        print("No audio files found in dataset.")
        return

    print(f"Loaded {len(X)} samples across {len(set(y.tolist()))} classes.")

    # Stratified split so each class is represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Fit scaler on training data only; transform both sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining K-Means (K={N_CLUSTERS})...")
    kmeans_model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )
    kmeans_model.fit(X_train_scaled)

    # Build cluster→label map using majority vote on training set
    train_cluster_ids = kmeans_model.predict(X_train_scaled)
    cluster_to_label = build_cluster_label_map(train_cluster_ids, y_train, N_CLUSTERS)

    # Predict on held-out test set
    test_cluster_ids = kmeans_model.predict(X_test_scaled)
    y_pred = [cluster_to_label.get(c, f"cluster_{c}") for c in test_cluster_ids]
    y_true = y_test.tolist()

    print(f"Total test samples evaluated: {len(y_true)}")

    all_labels = sorted(list(set(y_true + y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print(" MODEL EVALUATION METRICS  (hold-out test set)")
    print("=" * 60)
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print("\n" + "-" * 60)
    print(" PER-CLASS METRICS")
    print("-" * 60)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

    print(f"\n  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("  " + "-" * 66)
    for i, label in enumerate(all_labels):
        support = sum(1 for yv in y_true if yv == label)
        print(f"  {label:<20} {per_class_precision[i]:<12.4f} {per_class_recall[i]:<12.4f} {per_class_f1[i]:<12.4f} {support:<10}")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(12, 10))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=all_labels,
        yticklabels=all_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Prediction Rate'}
    )

    plt.title(f'K-Means Sound Classification Confusion Matrix\n'
              f'Accuracy: {accuracy*100:.2f}% | F1 Score: {f1:.4f}  (test set)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Confusion matrix saved to: {output_path}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=all_labels,
        yticklabels=all_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Count'}
    )

    plt.title(f'K-Means Sound Classification Confusion Matrix (Counts)\n'
              f'Total Test Samples: {len(y_true)}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path_counts = os.path.join(RESULTS_PATH, 'confusion_matrix_counts.png')
    plt.savefig(output_path_counts, dpi=150, bbox_inches='tight')
    print(f"  Counts matrix saved to: {output_path_counts}")

    print("\n" + "=" * 60)
    print(" Displaying confusion matrix visualization...")
    print("=" * 60)
    plt.show()


if __name__ == '__main__':
    main()
