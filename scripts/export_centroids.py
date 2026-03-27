import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CLUSTER_LABELS_PATH = os.path.join(MODELS_DIR, 'cluster_labels.npy')
LABELS_PATH = os.path.join(MODELS_DIR, 'label_classes.npy')
OUTPUT_PATH = os.path.join(BASE_DIR, 'esp32', 'esp32_edge', 'src', 'centroids.h')


def export_centroids():

    print("=" * 60)
    print(" Exporting K-Means Model to C++ Header")
    print("=" * 60)
    
    if not os.path.exists(KMEANS_PATH):
        print("\nError: Model not found. Run /api/train first.")
        return
    
    print("\nLoading model...")
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    cluster_to_label = np.load(CLUSTER_LABELS_PATH, allow_pickle=True).item()
    label_classes = np.load(LABELS_PATH, allow_pickle=True)
    
    centroids = kmeans.cluster_centers_
    n_clusters, n_features = centroids.shape
    
    print(f"  Clusters: {n_clusters}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {list(label_classes)}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write("// Auto-generated K-Means model parameters\n")
        f.write("// DO NOT EDIT MANUALLY\n\n")
        f.write("#ifndef CENTROIDS_H\n")
        f.write("#define CENTROIDS_H\n\n")
        f.write("#if defined(ARDUINO_ARCH_ESP32)\n")
        f.write("#include <pgmspace.h>\n")
        f.write("#define CENTROIDS_PROGMEM PROGMEM\n")
        f.write("#else\n")
        f.write("#define CENTROIDS_PROGMEM\n")
        f.write("#endif\n\n")
        
        f.write(f"#define N_CLUSTERS {n_clusters}\n")
        f.write(f"#define N_FEATURES {n_features}\n")
        f.write(f"#define N_CLASSES {len(label_classes)}\n\n")
        
        f.write("const float SCALER_MEAN[N_FEATURES] CENTROIDS_PROGMEM = {\n")
        for i, val in enumerate(scaler.mean_):
            f.write(f"    {val:.8f}f{',' if i < len(scaler.mean_)-1 else ''}\n")
        f.write("};\n\n")
        
        f.write("const float SCALER_STD[N_FEATURES] CENTROIDS_PROGMEM = {\n")
        for i, val in enumerate(scaler.scale_):
            f.write(f"    {val:.8f}f{',' if i < len(scaler.scale_)-1 else ''}\n")
        f.write("};\n\n")
        
        f.write("const float CENTROIDS[N_CLUSTERS][N_FEATURES] CENTROIDS_PROGMEM = {\n")
        for i, centroid in enumerate(centroids):
            f.write("    {")
            for j, val in enumerate(centroid):
                f.write(f"{val:.8f}f{',' if j < len(centroid)-1 else ''}")
            f.write(f"}}{',' if i < len(centroids)-1 else ''}\n")
        f.write("};\n\n")
        
        f.write("const char* CLUSTER_LABELS[N_CLUSTERS] = {\n")
        for i in range(n_clusters):
            label = cluster_to_label.get(i, f"cluster_{i}")
            f.write(f'    "{label}"{"," if i < n_clusters-1 else ""}\n')
        f.write("};\n\n")
        
        f.write("const char* CLASS_NAMES[N_CLASSES] = {\n")
        for i, label in enumerate(label_classes):
            f.write(f'    "{label}"{"," if i < len(label_classes)-1 else ""}\n')
        f.write("};\n\n")
        
        f.write("#endif // CENTROIDS_H\n")
    
    file_size = os.path.getsize(OUTPUT_PATH) / 1024
    
    print(f"\n✓ Export complete!")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size: {file_size:.2f} KB")
    
    centroid_memory = n_clusters * n_features * 4
    total_memory = centroid_memory + (n_features * 2 * 4)
    
    print(f"\n  ESP32 Memory Usage:")
    print(f"    Centroids: {centroid_memory / 1024:.2f} KB")
    print(f"    Total: {total_memory / 1024:.2f} KB")
    
    if total_memory > 100 * 1024:
        print("\n  ⚠ Warning: Model size is large for ESP32")
        print("    Consider reducing N_CLUSTERS in app.py")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    export_centroids()
