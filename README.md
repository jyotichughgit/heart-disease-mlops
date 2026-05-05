# heart-disease-mlops
# Heart Disease Prediction — MLOps Pipeline
## BITS Pilani | MLOps (S2-25_AMLCSZG523) | Assignment I

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Local Setup](#local-setup)
3. [GCP Account Setup (Step-by-Step)](#gcp-account-setup)
4. [GKE Cluster Creation](#gke-cluster-creation)
5. [GitHub Secrets Configuration](#github-secrets-configuration)
6. [Running the CI/CD Pipeline](#running-the-cicd-pipeline)
7. [Accessing the API](#accessing-the-api)
8. [Monitoring with Prometheus & Grafana](#monitoring)
9. [Manual Docker Testing](#manual-docker-testing)

---

## Project Structure

```
heart-disease-mlops/
├── notebooks/
│   └── 01_eda_and_training.ipynb     # EDA + training (Jupyter)
├── src/
│   ├── app.py                         # FastAPI model-serving API 
│   ├── train.py                       # Standalone training script
│   └── models/                        # Saved model + feature config
├── tests/
│   └── test_pipeline.py              # Pytest unit tests
├── docker/
│   └── Dockerfile                     # Multi-stage Docker build
├── k8s/
│   └── deployment.yaml               # GKE Deployment + Service + HPA 
├── monitoring/
│   └── prometheus-grafana.yaml       # Prometheus + Grafana on K8s 
├── scripts/
│   ├── download_data.py              # Dataset download script
│   └── validate_model.py             # Model performance gate
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml        # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/heart-disease-mlops.git
cd heart-disease-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python scripts/download_data.py

# 5. Run training
python src/train.py --data data/heart.csv --output src/models/

# 6. Start API locally
cd src 
uvicorn app:app --reload --port 8000

# 7. Test the API 
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# 8. Run unit tests
pytest tests/ -v --cov=src
```

---

## GCP Account Setup

### Step 1 — Create a GCP Account
1. Go to https://cloud.google.com
2. Click **"Get started for free"** → sign in with Google account
3. GCP gives **$300 free credits** for 90 days (enough for this assignment)
4. Enter billing details (you won't be charged within free tier limits)

### Step 2 — Create a GCP Project
```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
# Then initialize:
gcloud init

# Create a new project
gcloud projects create heart-disease-mlops-<your-id> \
  --name="Heart Disease MLOps"

# Set it as active project
gcloud config set project heart-disease-mlops-<your-id>

# Verify
gcloud config get-value project
```

### Step 3 — Enable Required APIs
```bash
gcloud services enable \
  container.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com
```
> This takes ~2 minutes. You can also enable via Console → APIs & Services → Library

### Step 4 — Create a Service Account for GitHub Actions
```bash
# Create service account
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

# Grant required roles
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Download the JSON key 
gcloud iam service-accounts keys create gcp-sa-key.json \
  --iam-account=github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com

echo "Key saved to gcp-sa-key.json — keep this file SECRET!"
```

---

## GKE Cluster Creation

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create GKE cluster (e2-standard-2 is cost-effective for this assignment)
gcloud container clusters create heart-disease-cluster \
  --project=$PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=4 \
  --enable-autorepair \
  --enable-autoupgrade \
  --disk-size=50GB

# Get credentials (connects kubectl to your cluster)
gcloud container clusters get-credentials heart-disease-cluster \
  --zone=us-central1-a --project=$PROJECT_ID

# Verify cluster is running
kubectl get nodes
kubectl cluster-info
```

> ⚠️ **Cost tip**: A 2-node e2-standard-2 cluster costs ~$2–3/day. Delete it when not in use:
> `gcloud container clusters delete heart-disease-cluster --zone=us-central1-a`

---

## GitHub Secrets Configuration

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these two secrets:

| Secret Name       | Value                                              |
|-------------------|----------------------------------------------------|
| `GCP_PROJECT_ID`  | Your GCP project ID (e.g., `heart-disease-mlops-xyz`) |
| `GCP_SA_KEY`      | **Entire contents** of `gcp-sa-key.json` (paste the JSON) |

```bash
# Get your project ID
gcloud config get-value project

# Get the SA key contents (copy this entire output into GitHub)
cat gcp-sa-key.json
```

> ⚠️ **Never commit `gcp-sa-key.json` to Git.** Add it to `.gitignore` immediately:
> ```bash
> echo "gcp-sa-key.json" >> .gitignore
> ```

---

## Running the CI/CD Pipeline

Once secrets are configured, the pipeline triggers automatically on every `git push` to `main`:

```bash
# Make any change and push
git add .
git commit -m "feat: initial MLOps pipeline"
git push origin main
```

### Pipeline stages (in order):
```
1. Lint       → flake8 + black + isort checks
2. Test       → pytest unit tests + coverage report
3. Train      → downloads data, trains model, validates AUC ≥ 0.80
4. Build/Push → builds Docker image, pushes to GCR 
5. Deploy     → deploys to GKE, runs smoke test
```

Monitor progress at: `https://github.com/<username>/heart-disease-mlops/actions`

---

## Accessing the API

After deployment, get the Load Balancer IP: 
```bash
kubectl get service heart-disease-api-service -n mlops
# Look for EXTERNAL-IP column
```

### API Endpoints

| Endpoint          | Method | Description                    |
|-------------------|--------|--------------------------------|
| `/`               | GET    | Root — API info                |
| `/health`         | GET    | Health check                   |
| `/predict`        | POST   | Single prediction              |
| `/predict/batch`  | POST   | Batch predictions              |
| `/model/info`     | GET    | Model metadata                 |
| `/metrics`        | GET    | Prometheus metrics             |
| `/docs`           | GET    | Swagger UI                     |

### Sample prediction request:
```bash
curl -X POST http://<EXTERNAL-IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'  
```

Expected response:
```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "confidence": 0.8423,
  "probability_disease": 0.8423,
  "probability_no_disease": 0.1577,
  "model_name": "random_forest",
  "timestamp": "2025-01-01T10:00:00"
}
```

---

## Monitoring

### Deploy Prometheus + Grafana to GKE:
```bash
kubectl apply -f monitoring/prometheus-grafana.yaml

# Get Grafana external IP
kubectl get service grafana-service -n mlops
