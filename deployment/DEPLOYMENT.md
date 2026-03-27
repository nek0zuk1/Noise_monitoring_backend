# ZonoTrack - Deploying to Render

This guide shows how to deploy your ZonoTrack FastAPI backend to Render.

## Prerequisites

- Trained K-Means model files:
  - `kmeans_model.joblib`
  - `scaler.joblib`
  - `centroids.npy`
  - `cluster_labels.npy`
  - `label_classes.npy`

## Deployment Steps

### 1. Create a GitHub Repository

```bash
cd ZonoTrack
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/zonotrack.git
git push -u origin main
```

### 2. Deploy to Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` configuration
5. Click **"Create Web Service"**

### 3. Upload Model Files

Since model files are large and shouldn't be in git:

**Option A: Upload via Render Dashboard**
1. Go to your service → **"Shell"** tab
2. Upload files using the file upload feature

**Option B: Use Render Disk**
1. Create a persistent disk in Render
2. Mount it to `/opt/render/project/src`
3. Upload model files to the disk

**Option C: Download from Cloud Storage**
Add to your `app.py` startup:
```python
import urllib.request
import os

MODEL_URL = "https://your-storage.com/kmeans_model.joblib"
if not os.path.exists(KMEANS_PATH):
    urllib.request.urlretrieve(MODEL_URL, KMEANS_PATH)
```

### 4. Get Your API URL

After deployment, Render will provide a URL like:
```
https://zonotrack.onrender.com
```

### 5. Test Your API

```bash
# Test health check
curl https://zonotrack.onrender.com/api/classes

# Test prediction (with audio file)
curl -X POST -F "file=@test_audio.wav" https://zonotrack.onrender.com/api/predict
```

## API Endpoints

Once deployed, your ESP32 can access:

- `POST /api/predict` - Classify audio + get decibels
- `POST /api/train` - Train model (if needed)
- `GET /api/classes` - List sound categories
- `GET /api/centroids` - Export centroids for edge computing
- `GET /api/algorithm` - View K-Means formulas

## Environment Variables (Optional)

Add in Render dashboard under **"Environment"**:

```
MAX_CONTENT_LENGTH=16777216
SPL_CALIBRATION_OFFSET_DB=<optional_db_offset>
```

## Troubleshooting

### Build Fails
- Check that `requirements.txt` includes all dependencies
- Verify Python version (3.11.0 recommended)

### Model Not Found
- Ensure model files are uploaded to the correct directory
- Check file paths in `app.py`

### Out of Memory
- Reduce `N_CLUSTERS` in `app.py` (currently 256)
- Upgrade to a larger Render plan

### Slow Cold Starts
- Render free tier sleeps after inactivity
- Upgrade to paid plan for always-on service
- Or use a cron job to ping your API every 10 minutes

## Next Steps

- Configure ESP32 to use your deployed API URL
- See `ESP32_README.md` for hardware integration
