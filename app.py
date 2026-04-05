"""
ZonoTrack K-Means Clustering REST API
Urban Sound Classification with Explicit Computation

K-Means Algorithm:
==================
1. Initialize k centroids randomly from data points
2. Assign each point to nearest centroid using Euclidean distance
3. Update centroids as mean of assigned points
4. Repeat until convergence

Distance Computation:
====================
Euclidean Distance: d(x, c) = sqrt(sum((x_i - c_i)^2))

Where:
- x = feature vector of audio sample
- c = centroid vector
- i = feature dimension index

Confidence Score:
================
confidence = exp(-min_distance / 10) * 100

Lower distance to centroid = higher confidence
"""

import os
import tempfile
import math
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash


# CONSTANTS
SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
N_CLUSTERS = 96

# Optional SPL calibration offset (dB). If unset, SPL is reported as estimate only.
SPL_CALIBRATION_OFFSET_DB = os.environ.get('SPL_CALIBRATION_OFFSET_DB')
if SPL_CALIBRATION_OFFSET_DB is not None:
    try:
        SPL_CALIBRATION_OFFSET_DB = float(SPL_CALIBRATION_OFFSET_DB)
    except ValueError:
        SPL_CALIBRATION_OFFSET_DB = None

# Base directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# MONGODB SETUP
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://olsenparaiso4_db_user:paradise10032003@noisemonitoring.chvtlws.mongodb.net/?appName=NoiseMonitoring")
mongo_client = None
db = None
predictions_collection = None
users_collection = None
sensors_collection = None
sensor_readings_collection = None
reports_collection = None
admin_account_collection = None

FIXED_ADMIN_DOC_ID = 'fixed_admin_account'
FIXED_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = os.environ.get('ADMIN_DEFAULT_PASSWORD', 'admin123')


def init_mongo():
    global mongo_client, db, predictions_collection, users_collection, sensors_collection, sensor_readings_collection, reports_collection, admin_account_collection
    try:
        print("Connecting to MongoDB...")
        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
        )
        # Force a real connection check so failures are caught immediately.
        print("Pinging MongoDB...")
        mongo_client.admin.command('ping')
        print("MongoDB connected successfully!")
        db = mongo_client['NoiseMonitoring_db']
        predictions_collection = db['predictions']
        users_collection = db['users']
        sensors_collection = db['sensors']
        sensor_readings_collection = db['sensor_readings']
        reports_collection = db['reports']
        admin_account_collection = db['admin_account']
        users_collection.create_index('username', unique=True)
        admin_account_collection.create_index('username', unique=True)
        sensor_readings_collection.create_index([('sensor_id', 1), ('received_at', -1)])
        reports_collection.create_index([('created_at', -1)])
        normalize_user_schema()
        ensure_fixed_admin_account()
        print("MongoDB initialization complete!")
        return True, None
    except (Exception, KeyboardInterrupt) as e:
        print(f"MongoDB connection FAILED: {e}")
        mongo_client = None
        db = None
        predictions_collection = None
        users_collection = None
        sensors_collection = None
        sensor_readings_collection = None
        reports_collection = None
        admin_account_collection = None
        return False, str(e)


def ensure_mongo_ready():
    if mongo_client is not None and users_collection is not None:
        try:
            mongo_client.admin.command('ping')
            return True, None
        except Exception:
            pass
    return init_mongo()


def normalize_user_schema():
    try:
        users_collection.update_many({}, {'$unset': {'role': ''}})
        users_collection.delete_many({'username': FIXED_ADMIN_USERNAME})
    except Exception as e:
        print(f"Warning: failed to normalize user schema: {e}")


def ensure_fixed_admin_account():
    existing = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})
    if existing:
        return

    admin_account_collection.insert_one({
        '_id': FIXED_ADMIN_DOC_ID,
        'username': FIXED_ADMIN_USERNAME,
        'holder_name': 'Administrator',
        'password': generate_password_hash(DEFAULT_ADMIN_PASSWORD),
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'active': True,
    })


# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CENTROIDS_PATH = os.path.join(MODELS_DIR, 'centroids.npy')
CLUSTER_LABELS_PATH = os.path.join(MODELS_DIR, 'cluster_labels.npy')
LABELS_PATH = os.path.join(MODELS_DIR, 'label_classes.npy')
CLASS_PROTOTYPES_PATH = os.path.join(MODELS_DIR, 'class_prototypes.npy')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test_data')

# Global model variables
kmeans_model = None
scaler = None
cluster_to_label = None
label_classes = None
class_prototypes = None


# COMPUTATION FUNCTIONS
def euclidean_distance(x, centroid):
    squared_diff = 0.0
    for i in range(len(x)):
        diff = x[i] - centroid[i]
        squared_diff += diff * diff

    distance = math.sqrt(squared_diff)
    return distance


def compute_all_distances(x, centroids):
    distances = []
    for centroid in centroids:
        d = euclidean_distance(x, centroid)
        distances.append(d)
    return distances


def find_nearest_cluster(distances):
    min_distance = distances[0]
    min_index = 0

    for i in range(1, len(distances)):
        if distances[i] < min_distance:
            min_distance = distances[i]
            min_index = i

    return min_index, min_distance


def compute_confidence(min_distance):
    confidence = math.exp(-min_distance / 10.0) * 100.0
    return min(confidence, 99.9)

# Compute RMS (Root Mean Square)
def compute_decibels(signal):
    rms = math.sqrt(sum(s * s for s in signal) / len(signal))

    # Avoid log(0)
    if rms < 1e-10:
        rms = 1e-10

    # dBFS (decibels relative to full scale)
    db_fs = 20 * math.log10(rms)

    # Baseline SPL estimate for relative monitoring only.
    spl_estimate = db_fs + 94

    spl_calibrated = None
    if SPL_CALIBRATION_OFFSET_DB is not None:
        spl_calibrated = db_fs + SPL_CALIBRATION_OFFSET_DB

    spl_value_for_indicator = spl_calibrated if spl_calibrated is not None else spl_estimate
    period = get_noise_period()

    return {
        'rms': round(rms, 6),
        'db_fs': round(db_fs, 2),
        'spl_estimated_db': round(max(0, spl_estimate), 2),
        'spl_calibrated_db': round(max(0, spl_calibrated), 2) if spl_calibrated is not None else None,
        'spl_calibration_offset_db': SPL_CALIBRATION_OFFSET_DB,
        'spl_is_calibrated': spl_calibrated is not None,
        'spl_value_db': round(max(0, spl_value_for_indicator), 2),
        'noise_period': period,
        'formula': 'dB = 20 * log10(RMS)',
        'description': get_noise_level_description(spl_value_for_indicator, period=period)
    }


def get_noise_period(hour=None):
    if hour is None:
        hour = datetime.now().hour

    if 6 <= hour < 18:
        return 'morning'
    return 'evening'


def get_noise_level_description(db_spl, period='morning'):
    if period == 'evening':
        # WHO-style evening interpretation: lower comfort target than daytime.
        if db_spl < 40:
            return 'Very quiet evening environment'
        elif db_spl < 48:
            return 'Evening level within WHO-oriented target range'
        elif db_spl < 60:
            return 'Slightly elevated for evening comfort'
        elif db_spl < 70:
            return 'High evening noise; may disturb rest'
        elif db_spl < 85:
            return 'Very high evening noise; reduce exposure'
        else:
            return 'Potentially harmful if prolonged'

    # Morning/daytime profile using WHO-oriented reference around 53 dB (road traffic context).
    if db_spl < 40:
        return 'Very quiet morning environment'
    elif db_spl < 53:
        return 'Morning level within WHO-oriented target range'
    elif db_spl < 65:
        return 'Moderately elevated morning noise'
    elif db_spl < 75:
        return 'High morning noise exposure'
    elif db_spl < 85:
        return 'Very high noise; limit duration'
    else:
        return 'Potentially harmful if prolonged'


def get_noise_indicator(db_spl):
    if db_spl < 55:
        return 'Normal'
    elif db_spl < 70:
        return 'Elevated'
    else:
        return 'High'


def to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_mfcc_delta(mfccs, order=1, default_width=9):
    n_frames = mfccs.shape[1]
    max_valid_width = n_frames if n_frames % 2 == 1 else n_frames - 1

    if max_valid_width < 3:
        return np.zeros_like(mfccs)

    width = min(default_width, max_valid_width)
    if width % 2 == 0:
        width -= 1

    if width < 3:
        return np.zeros_like(mfccs)

    return librosa.feature.delta(mfccs, order=order, width=width, mode='interp')


# FEATURE EXTRACTION
def extract_mfcc_features(file_path, return_signal=False):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=NUM_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Add temporal dynamics and spectral descriptors for better class separation.
    mfcc_delta = safe_mfcc_delta(mfccs, order=1)
    mfcc_delta2 = safe_mfcc_delta(mfccs, order=2)

    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=HOP_LENGTH)

    features = np.concatenate([
        mfccs.mean(axis=1),
        mfccs.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta.std(axis=1),
        mfcc_delta2.mean(axis=1),
        mfcc_delta2.std(axis=1),
        np.array([
            spectral_centroid.mean(), spectral_centroid.std(),
            spectral_bandwidth.mean(), spectral_bandwidth.std(),
            spectral_rolloff.mean(), spectral_rolloff.std(),
            zcr.mean(), zcr.std(),
        ])
    ])

    if return_signal:
        return features, signal
    return features


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_dataset_class_names():
    if not os.path.exists(DATASET_PATH):
        return []

    classes = [
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]
    return sorted(classes)


def load_model(): 
    global kmeans_model, scaler, cluster_to_label, label_classes, class_prototypes

    if os.path.exists(KMEANS_PATH) and os.path.exists(SCALER_PATH):
        kmeans_model = joblib.load(KMEANS_PATH)
        scaler = joblib.load(SCALER_PATH)

        if os.path.exists(CLUSTER_LABELS_PATH):
            cluster_to_label = np.load(CLUSTER_LABELS_PATH, allow_pickle=True).item()
        if os.path.exists(LABELS_PATH):
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
        if os.path.exists(CLASS_PROTOTYPES_PATH):
            class_prototypes = np.load(CLASS_PROTOTYPES_PATH, allow_pickle=True).item()

        # Keep API class list aligned with dataset folders.
        dataset_classes = get_dataset_class_names()
        if dataset_classes:
            label_classes = np.array(dataset_classes)

        print("K-Means model loaded")
    else:
        print("Model not found. Run /api/train first.")


def predict_by_class_prototype(features_scaled):
    if not class_prototypes:
        return None, None

    best_label = None
    best_distance = None

    for label, prototype in class_prototypes.items():
        d = euclidean_distance(features_scaled, prototype)
        if best_distance is None or d < best_distance:
            best_distance = d
            best_label = label

    return best_label, best_distance


# FASTAPI APP SETUP with lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup
    mongo_ok, mongo_error = init_mongo()
    if not mongo_ok:
        print(f"MongoDB initialization error: {mongo_error}")
    load_model()

    print("\n" + "="*50)
    print("ZonoTrack K-Means Clustering API")
    print("="*50)
    print("\nAlgorithm: K-Means Clustering")
    print("Distance Formula: d = sqrt(sum((x_i - c_i)^2))")
    print("Decibel Formula: dB = 20 * log10(RMS)")
    print("\nEndpoints:")
    print("  POST /api/predict   - Classify audio (with dB measurement)")
    print("  POST /api/train     - Train K-Means model")
    print("  GET  /api/classes   - List categories")
    print("  GET  /api/centroids - Export for ESP32")
    print("  GET  /api/algorithm       - View formulas")
    print("  GET  /api/history         - View recent predictions (MongoDB)")
    print("  POST /api/auth/login      - Authenticate admin/user account")
    print("  POST /api/admin/create_user - Create a user account")
    print("  GET  /api/admin/users       - List all user accounts")
    print("  PUT/DELETE /api/admin/users/<id> - Update or delete users")
    print("  GET/PUT /api/admin/account  - View/update fixed admin account")
    print("  GET/POST /api/sensors       - Manage SensorData (matching Frontend)")
    print("  GET /api/sensors/<id>/readings - Sensor reading history")
    print("  POST /api/reports - Submit client report/proof")
    print("  GET /api/admin/reports/summary - Admin reports summary")
    print("  GET /api/admin/noise-summary?window=24h|7d - Admin graph summaries")
    print("  DELETE /api/sensors/<id>    - Delete SensorData")
    print("\nServer URL:")
    print("  Local:   http://127.0.0.1:5000")
    print("  Network: http://0.0.0.0:5000")
    print("="*50 + "\n")

    yield

    # Shutdown
    if mongo_client is not None:
        mongo_client.close()


app = FastAPI(title="ZonoTrack K-Means Clustering API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler to return {"error": "..."} matching the frontend expectation
# (FastAPI default is {"detail": "..."} which the frontend does not read)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={'error': exc.detail},
    )


# API ENDPOINTS
@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if kmeans_model is None:
        raise HTTPException(status_code=500, detail='Model not loaded. Run /api/train first.')

    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail='Invalid file. Only .wav, .mp3, and .m4a are allowed.')

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_CONTENT_LENGTH:
        raise HTTPException(status_code=400, detail='File too large. Max 16 MB.')

    filepath = None
    try:
        # Add timestamp prefix to avoid collisions from concurrent uploads
        filename = secure_filename(file.filename)
        unique_filename = f"{int(datetime.utcnow().timestamp() * 1000)}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(filepath, 'wb') as f:
            f.write(contents)

        # Extract features and signal for dB calculation
        features, signal = extract_mfcc_features(filepath, return_signal=True)
        features_scaled = scaler.transform([features])[0]

        # COMPUTATION: Calculate decibels
        decibels = compute_decibels(signal)

        dataset_classes = set(get_dataset_class_names())
        if not dataset_classes:
            raise HTTPException(status_code=500, detail='No dataset categories found')

        # Prefer class-prototype prediction (supervised labels) for better accuracy.
        predicted_class, min_distance = predict_by_class_prototype(features_scaled)
        if predicted_class not in dataset_classes:
            predicted_class = None

        # Fallback to nearest K-Means cluster label if prototypes are unavailable.
        if predicted_class is None:
            centroids = kmeans_model.cluster_centers_
            distances = compute_all_distances(features_scaled, centroids)
            cluster_idx, min_distance = find_nearest_cluster(distances)
            if cluster_to_label and cluster_idx in cluster_to_label:
                candidate_class = cluster_to_label[cluster_idx]
                if candidate_class in dataset_classes:
                    predicted_class = candidate_class

        if predicted_class is None:
            raise HTTPException(status_code=500, detail='Predicted cluster is not mapped to a valid dataset category. Run /api/train to sync model.')

        # COMPUTATION: Calculate confidence from nearest label/prototype distance.
        confidence = compute_confidence(min_distance)

        # Save to MongoDB
        try:
            if mongo_client is not None:
                record = {
                    'filename': filename,
                    'class': predicted_class,
                    'confidence': round(confidence, 2),
                    'decibels': round(decibels['spl_value_db'], 2),
                    'spl_estimated_db': decibels['spl_estimated_db'],
                    'spl_calibrated_db': decibels['spl_calibrated_db'],
                    'spl_is_calibrated': decibels['spl_is_calibrated'],
                    'spl_calibration_offset_db': decibels['spl_calibration_offset_db'],
                    'indicator': get_noise_indicator(decibels['spl_value_db']),
                    'timestamp': datetime.utcnow()
                }
                predictions_collection.insert_one(record)
        except Exception as e:
            print(f"Warning: Failed to log prediction to MongoDB: {e}")

        return {
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'decibels': round(decibels['spl_value_db'], 2),
            'indicator': get_noise_indicator(decibels['spl_value_db'])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temp file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.post('/api/train')
async def train():
    global kmeans_model, scaler, cluster_to_label, label_classes, class_prototypes

    try:
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=400, detail='Dataset folder not found')

        dataset_classes = get_dataset_class_names()
        allowed_classes = set(dataset_classes)

        features_list = []
        labels_list = []

        for root, dirs, files in os.walk(DATASET_PATH):
            for f in files:
                if f.endswith(('.wav', '.mp3', '.m4a')):
                    file_path = os.path.join(root, f)
                    label = os.path.basename(root)
                    if label not in allowed_classes:
                        continue

                    try:
                        features = extract_mfcc_features(file_path)
                        features_list.append(features)
                        labels_list.append(label)
                    except Exception as e:
                        print(f"Error: {file_path}: {e}")

        if len(features_list) == 0:
            raise HTTPException(status_code=400, detail='No audio files found')

        X = np.array(features_list)
        y = np.array(labels_list)

        # Scale features (standardization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(features_list)
        effective_n_clusters = min(N_CLUSTERS, n_samples)

        # Train K-Means
        kmeans_model = KMeans(
            n_clusters=effective_n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        kmeans_model.fit(X_scaled)

        # Map clusters to labels (majority voting)
        cluster_assignments = kmeans_model.predict(X_scaled)
        cluster_to_label = {}

        for cluster_id in range(effective_n_clusters):
            mask = cluster_assignments == cluster_id
            if np.any(mask):
                cluster_labels_arr = y[mask]
                most_common = Counter(cluster_labels_arr).most_common(1)
                if most_common:
                    cluster_to_label[cluster_id] = most_common[0][0]

        # Supervised class prototypes in scaled feature space.
        class_prototypes = {}
        for label in dataset_classes:
            label_mask = y == label
            if np.any(label_mask):
                class_prototypes[label] = X_scaled[label_mask].mean(axis=0)

        label_classes = np.array(dataset_classes)

        # Save model
        joblib.dump(kmeans_model, KMEANS_PATH)
        joblib.dump(scaler, SCALER_PATH)
        np.save(CENTROIDS_PATH, kmeans_model.cluster_centers_)
        np.save(CLUSTER_LABELS_PATH, cluster_to_label)
        np.save(LABELS_PATH, label_classes)
        np.save(CLASS_PROTOTYPES_PATH, class_prototypes)

        # Calculate inertia (sum of squared distances)
        inertia = kmeans_model.inertia_

        return {
            'status': 'completed',
            'samples': len(features_list),
            'classes': label_classes.tolist(),
            'n_clusters_requested': N_CLUSTERS,
            'n_clusters_used': effective_n_clusters,
            'computation': {
                'algorithm': 'K-Means Clustering',
                'distance_metric': 'Euclidean: d = sqrt(sum((x_i - c_i)^2))',
                'centroid_update': 'Mean: c = (1/n) * sum(x_i)',
                'prediction_strategy': 'Nearest class prototype (fallback: nearest K-Means cluster label)',
                'inertia': round(float(inertia), 2),
                'iterations': kmeans_model.n_iter_
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/classes')
async def get_classes():
    classes_from_dataset = get_dataset_class_names()

    if classes_from_dataset:
        return {'classes': classes_from_dataset}

    if label_classes is None:
        raise HTTPException(status_code=500, detail='Labels not loaded')

    return {'classes': label_classes.tolist()}


@app.get('/api/centroids')
async def get_centroids():
    if kmeans_model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    return {
        'centroids': kmeans_model.cluster_centers_.tolist(),
        'cluster_labels': cluster_to_label if cluster_to_label else {},
        'n_clusters': int(kmeans_model.n_clusters),
        'n_clusters_requested': N_CLUSTERS,
        'feature_dim': kmeans_model.cluster_centers_.shape[1]
    }


@app.get('/api/algorithm')
async def get_algorithm_info():
    return {
        'algorithm': 'K-Means Clustering',
        'steps': [
            '1. Initialize k centroids randomly',
            '2. Assign each point to nearest centroid',
            '3. Update centroids as mean of points',
            '4. Repeat until convergence'
        ],
        'formulas': {
            'euclidean_distance': 'd(x, c) = sqrt(sum((x_i - c_i)^2))',
            'centroid_update': 'c_new = (1/n) * sum(x_i)',
            'class_prototype': 'label* = argmin_label d(x, prototype_label)',
            'confidence': 'confidence = exp(-distance/10) * 100',
            'db_fs': 'dBFS = 20 * log10(RMS)',
            'spl_estimated': 'SPL_estimated = dBFS + 94 (heuristic baseline)',
            'spl_calibrated': 'SPL_calibrated = dBFS + SPL_CALIBRATION_OFFSET_DB (when provided)'
        },
        'parameters': {
            'n_clusters': N_CLUSTERS,
            'max_iterations': 300,
            'n_init': 10,
            'decibel_formula': 'dB = 20 * log10(RMS)',
            'spl_calibration_offset_db': SPL_CALIBRATION_OFFSET_DB
        }
    }


@app.get('/api/history')
async def get_history():
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        cursor = predictions_collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(50)
        history = list(cursor)
        # Convert datetime objects to ISO strings for JSON serialization
        for item in history:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
        return {'history': history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/auth/login')
async def auth_login(request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    data = await request.json()
    if not data or not data.get('username') or not data.get('password'):
        raise HTTPException(status_code=400, detail='username and password are required')

    username = str(data.get('username')).strip().lower()
    password = str(data.get('password'))

    if username == FIXED_ADMIN_USERNAME:
        admin_doc = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})
        if not admin_doc:
            ensure_fixed_admin_account()
            admin_doc = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})

        if not admin_doc or not check_password_hash(admin_doc.get('password', ''), password):
            raise HTTPException(status_code=401, detail='Invalid username or password')

        return {
            'message': 'Login successful',
            'user': {
                'id': FIXED_ADMIN_DOC_ID,
                'name': admin_doc.get('holder_name', 'Administrator'),
                'username': FIXED_ADMIN_USERNAME,
                'email': 'admin@admin.com',
                'is_admin': True,
            }
        }

    user_doc = users_collection.find_one({'username': username})
    if not user_doc:
        raise HTTPException(status_code=401, detail='Invalid username or password')

    if not check_password_hash(user_doc.get('password', ''), password):
        raise HTTPException(status_code=401, detail='Invalid username or password')

    return {
        'message': 'Login successful',
        'user': {
            'id': str(user_doc.get('_id')),
            'name': user_doc.get('name', username),
            'username': user_doc.get('username', username),
            'email': f"{user_doc.get('username', username)}@user.com",
            'is_admin': False,
        }
    }

#Endpoint for admin to create a new user account (users only).
@app.post('/api/admin/create_user', status_code=201)
async def create_user(request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    data = await request.json()

    if not data or not data.get('username') or not data.get('password') or not data.get('name'):
        raise HTTPException(status_code=400, detail='Missing required fields: name, username, password')

    name = data.get('name')
    username = data.get('username').strip().lower()
    password = data.get('password')

    if len(password) < 6:
        raise HTTPException(status_code=400, detail='Password must be at least 6 characters')

    if username == FIXED_ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail='Username admin is reserved for fixed admin account')

    # Check if user already exists
    try:
        if users_collection.find_one({'username': username}):
            raise HTTPException(status_code=400, detail='Username already exists')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'MongoDB query failed: {e}')

    # Hash the password for security
    hashed_password = generate_password_hash(password)

    # Create the user document (record/row)
    new_user = {
        'name': name,
        'username': username,
        'password': hashed_password,
        'created_at': datetime.utcnow(),
        'active': True
    }

    try:
        users_collection.insert_one(new_user)
        return {
            'message': 'User created successfully',
            'user': {
                'name': name,
                'username': username,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'MongoDB insert failed: {e}')

#Retrieve all users.
@app.get('/api/admin/users')
async def get_users():
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')
    try:
        users = list(users_collection.find({}))
        for user in users:
            user['_id'] = str(user['_id'])
            # Don't send password hash to frontend
            user.pop('password', None)
            # Convert datetime objects to ISO strings
            for key, value in user.items():
                if isinstance(value, datetime):
                    user[key] = value.isoformat()
        return {'users': users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update a regular user.
@app.put('/api/admin/users/{user_id}')
async def update_user(user_id: str, request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user ID format')

    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail='No data provided')

    update_fields = {}
    if 'name' in data:
        update_fields['name'] = data['name']
    if 'username' in data:
        next_username = data['username'].strip().lower()
        if next_username == FIXED_ADMIN_USERNAME:
            raise HTTPException(status_code=400, detail='Username admin is reserved')
        update_fields['username'] = next_username
    if 'password' in data and data['password']:
        update_fields['password'] = generate_password_hash(data['password'])

    if not update_fields:
        return {'message': 'No changes detected'}

    try:
        result = users_collection.update_one({'_id': obj_id}, {'$set': update_fields})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail='User not found')
        return {'message': 'User updated successfully'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete a regular user.
@app.delete('/api/admin/users/{user_id}')
async def delete_user(user_id: str):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user ID format')

    try:
        target = users_collection.find_one({'_id': obj_id}, {'username': 1})
        if target and target.get('username') == FIXED_ADMIN_USERNAME:
            raise HTTPException(status_code=400, detail='Fixed admin account cannot be deleted from users')
        result = users_collection.delete_one({'_id': obj_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail='User not found')
        return {'message': 'User deleted successfully'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get or update the fixed admin account details (password and holder name).
@app.get('/api/admin/account')
async def get_fixed_admin_account():
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    account = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})
    if not account:
        ensure_fixed_admin_account()
        account = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})

    updated_at = account.get('updated_at')
    updated_at_value = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else updated_at
    return {
        'admin': {
            'username': account.get('username', FIXED_ADMIN_USERNAME),
            'holder_name': account.get('holder_name', 'Administrator'),
            'updated_at': updated_at_value,
        }
    }

# Only update the fixed admin account password and holder name (no deletion allowed).
@app.put('/api/admin/account')
async def update_fixed_admin_account(request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    account = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})
    if not account:
        ensure_fixed_admin_account()
        account = admin_account_collection.find_one({'_id': FIXED_ADMIN_DOC_ID})

    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail='No data provided')

    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    holder_name = data.get('holder_name', '').strip()

    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail='current_password and new_password are required')

    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail='new_password must be at least 6 characters')

    if not check_password_hash(account.get('password', ''), current_password):
        raise HTTPException(status_code=401, detail='Current admin password is incorrect')

    update_doc = {
        'password': generate_password_hash(new_password),
        'updated_at': datetime.utcnow(),
    }

    if holder_name:
        update_doc['holder_name'] = holder_name

    admin_account_collection.update_one(
        {'_id': FIXED_ADMIN_DOC_ID},
        {'$set': update_doc}
    )

    return {'message': 'Fixed admin account updated successfully'}

# Retrieve all sensors.
@app.get('/api/sensors')
async def get_sensors():
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        cursor = sensors_collection.find({}, {'_id': 0})
        sensors = list(cursor)
        # Convert datetime objects to ISO strings
        for sensor in sensors:
            for key, value in sensor.items():
                if isinstance(value, datetime):
                    sensor[key] = value.isoformat()
        return {'sensors': sensors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create or update sensor data (upsert by custom string ID).
@app.post('/api/sensors')
async def push_sensor_data(request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    data = await request.json()
    if not data:
        data = {}

    sensor_id = str(data.get('id') or data.get('sensorId') or '').strip()
    if not sensor_id:
        raise HTTPException(status_code=400, detail='Missing required sensor field: id')

    sensor_name = str(data.get('name') or f'Sensor {sensor_id}').strip()
    noise_level = round(to_float(data.get('noiseLevel', 0), 0.0), 2)
    now = datetime.utcnow()
    upload_interval_sec = max(1, to_int(data.get('uploadIntervalSec', 5), 5))

    sensor_doc = {
        'id': sensor_id,
        'name': sensor_name,
        'status': data.get('status', 'inactive'),
        'noiseLevel': noise_level,
        'class': data.get('class', 'Unknown'),
        'confidence': round(to_float(data.get('confidence', 0), 0.0), 2),
        'decibels': round(to_float(data.get('decibels', noise_level), noise_level), 2),
        'indicator': data.get('indicator', get_noise_indicator(noise_level)),
        'location': data.get('location', ''),
        'lastUpdate': data.get('lastUpdate') or now.isoformat() + 'Z',
        'uploadIntervalSec': upload_interval_sec,
        'updated_at': now
    }

    reading_doc = {
        'sensor_id': sensor_id,
        'name': sensor_name,
        'noiseLevel': noise_level,
        'class': sensor_doc['class'],
        'confidence': sensor_doc['confidence'],
        'decibels': sensor_doc['decibels'],
        'indicator': sensor_doc['indicator'],
        'status': sensor_doc['status'],
        'location': sensor_doc['location'],
        'source': data.get('source', 'sensor'),
        'received_at': now,
    }

    try:
        # Upsert the sensor by custom string ID
        sensors_collection.update_one(
            {'id': sensor_doc['id']},
            {'$set': sensor_doc},
            upsert=True
        )

        # Keep per-upload history so readings sent every 5 seconds are stored.
        if sensor_readings_collection is not None:
            sensor_readings_collection.insert_one(reading_doc)

        # Convert datetime for JSON response
        response_doc = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in sensor_doc.items()}

        return {
            'message': 'Sensor data recorded successfully',
            'sensor': response_doc,
            'nextUploadInSec': upload_interval_sec,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieve historical readings for one sensor, newest first, with optional limit (default 120 for 10 minutes at 5-sec intervals).
@app.get('/api/sensors/{sensor_id}/readings')
async def get_sensor_readings(sensor_id: str, limit: int = Query(default=120, ge=1, le=1000)):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        cursor = sensor_readings_collection.find(
            {'sensor_id': sensor_id},
            {'_id': 0}
        ).sort('received_at', -1).limit(limit)
        readings = list(cursor)
        # Convert datetime objects to ISO strings
        for reading in readings:
            for key, value in reading.items():
                if isinstance(value, datetime):
                    reading[key] = value.isoformat()
        return {'sensor_id': sensor_id, 'readings': readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Receive client proof/report submissions for admin review.
@app.post('/api/reports', status_code=201)
async def submit_client_report(request: Request):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    data = await request.json()
    if not data:
        data = {}

    report_text = str(data.get('report_text', '')).strip()
    image_uri = str(data.get('image_uri', '')).strip()
    submitted_by = str(data.get('submitted_by', 'mobile_client')).strip() or 'mobile_client'
    location = str(data.get('location', '')).strip()

    if not report_text and not image_uri:
        raise HTTPException(status_code=400, detail='At least one of report_text or image_uri is required')

    report_doc = {
        'report_text': report_text,
        'image_uri': image_uri,
        'submitted_by': submitted_by,
        'location': location,
        'status': 'pending',
        'created_at': datetime.utcnow(),
    }

    try:
        result = reports_collection.insert_one(report_doc)
        return {
            'message': 'Report submitted successfully',
            'report_id': str(result.inserted_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoint to get summary metrics and recent client reports for dashboard display.
@app.get('/api/admin/reports/summary')
async def get_admin_report_summary():
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        total_reports = reports_collection.count_documents({})
        pending_reports = reports_collection.count_documents({'status': 'pending'})
        reports_today = reports_collection.count_documents({
            'created_at': {
                '$gte': datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            }
        })

        recent_cursor = reports_collection.find({}, {'_id': 0}).sort('created_at', -1).limit(10)
        recent_reports = list(recent_cursor)
        # Convert datetime objects to ISO strings
        for report in recent_reports:
            for key, value in report.items():
                if isinstance(value, datetime):
                    report[key] = value.isoformat()

        return {
            'summary': {
                'total_reports': total_reports,
                'pending_reports': pending_reports,
                'reports_today': reports_today,
            },
            'recent_reports': recent_reports,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoint to get graph-ready noise summaries for the last 24 hours or 7 days.
@app.get('/api/admin/noise-summary')
async def get_admin_noise_summary(window: str = Query(default='24h')):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    window = window.strip().lower()
    if window not in ('24h', '7d'):
        raise HTTPException(status_code=400, detail='window must be either 24h or 7d')

    now = datetime.utcnow()
    hours = 24 if window == '24h' else 24 * 7
    since = now - timedelta(hours=hours)

    try:
        cursor = sensor_readings_collection.find(
            {'received_at': {'$gte': since}},
            {'_id': 0, 'received_at': 1, 'decibels': 1, 'indicator': 1}
        )
        readings = list(cursor)

        if window == '24h':
            total_buckets = 24
            bucket_starts = [
                (now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
                for offset in range(total_buckets - 1, -1, -1)
            ]
            label_for = lambda dt: dt.strftime('%H:00')
            bucket_key_for_reading = lambda dt: dt.replace(minute=0, second=0, microsecond=0)
        else:
            total_buckets = 7
            bucket_starts = [
                (now - timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
                for offset in range(total_buckets - 1, -1, -1)
            ]
            label_for = lambda dt: dt.strftime('%a')
            bucket_key_for_reading = lambda dt: dt.replace(hour=0, minute=0, second=0, microsecond=0)

        bucket_map = {
            bucket_start: {'sum_db': 0.0, 'count': 0}
            for bucket_start in bucket_starts
        }

        indicator_counts = {'Normal': 0, 'Elevated': 0, 'High': 0}
        total_db = 0.0
        valid_db_count = 0

        for reading in readings:
            received_at = reading.get('received_at')
            if not isinstance(received_at, datetime):
                continue

            decibels = to_float(reading.get('decibels', 0), 0.0)
            key = bucket_key_for_reading(received_at)

            if key in bucket_map:
                bucket_map[key]['sum_db'] += decibels
                bucket_map[key]['count'] += 1

            total_db += decibels
            valid_db_count += 1

            indicator = str(reading.get('indicator', '')).strip()
            if indicator in indicator_counts:
                indicator_counts[indicator] += 1

        graph = []
        for bucket_start in bucket_starts:
            bucket = bucket_map[bucket_start]
            count = bucket['count']
            avg_db = round((bucket['sum_db'] / count), 2) if count > 0 else 0.0
            graph.append({
                'label': label_for(bucket_start),
                'avg_decibels': avg_db,
                'count': count,
            })

        average_decibels = round((total_db / valid_db_count), 2) if valid_db_count > 0 else 0.0

        return {
            'window': window,
            'average_decibels': average_decibels,
            'total_readings': valid_db_count,
            'noise_summary': indicator_counts,
            'graph': graph,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoint to delete a sensor by its custom string ID and all its associated readings.
@app.delete('/api/sensors/{sensor_id}')
async def delete_sensor(sensor_id: str):
    mongo_ok, mongo_error = ensure_mongo_ready()
    if not mongo_ok:
        raise HTTPException(status_code=500, detail=f'MongoDB connection failed: {mongo_error}')

    try:
        result = sensors_collection.delete_one({'id': sensor_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail='Sensor not found')
        # Also remove orphaned readings for this sensor
        if sensor_readings_collection is not None:
            sensor_readings_collection.delete_many({'sensor_id': sensor_id})
        return {'message': 'Sensor deleted successfully'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility function to find an available port for the server to listen on (default starting at 5000).
def find_available_port(start_port=5000, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")


if __name__ == '__main__':
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000)) 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
