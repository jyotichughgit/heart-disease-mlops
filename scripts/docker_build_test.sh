#!/bin/bash
# =============================================================================
# scripts/docker_build_test.sh
# Builds and tests the Docker container locally for Task 6.
# Usage: bash scripts/docker_build_test.sh
# =============================================================================

set -e

IMAGE_NAME="heart-disease-api:local"
CONTAINER_NAME="heart-disease-test"
PORT=8000

echo "============================================"
echo " Heart Disease API - Docker Build & Test"
echo "============================================"

# ── Step 1: Check model files exist ──────────────────────────────────────────
echo ""
echo "[1/5] Checking model files..."
if [ ! -f "src/models/best_model.pkl" ]; then
    echo "  Model not found. Training model first..."
    python scripts/download_data.py
    python src/train.py --data data/heart.csv --output src/models/
fi
ls -lh src/models/
echo "  Model files ready ✓"

# ── Step 2: Build Docker image ────────────────────────────────────────────────
echo ""
echo "[2/5] Building Docker image..."
docker build \
    -f docker/Dockerfile \
    -t $IMAGE_NAME \
    .
echo "  Image built: $IMAGE_NAME ✓"

# ── Step 3: Stop any existing container ───────────────────────────────────────
echo ""
echo "[3/5] Starting container..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Mount models folder and set correct paths
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -v "$(pwd)/src/models:/app/models" \
    -e MODEL_PATH=/app/models/best_model.pkl \
    -e CONFIG_PATH=/app/models/feature_config.json \
    $IMAGE_NAME

echo "  Container started: $CONTAINER_NAME ✓"

# ── Step 4: Wait for API to be ready ─────────────────────────────────────────
echo ""
echo "[4/5] Waiting for API to be ready..."
max_retries=20
count=0
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
    count=$((count + 1))
    if [ $count -ge $max_retries ]; then
        echo "  API did not start. Container logs:"
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME
        exit 1
    fi
    echo "  Waiting... ($count/$max_retries)"
    sleep 2
done
echo "  API is ready ✓"

# ── Step 5: Test endpoints ────────────────────────────────────────────────────
echo ""
echo "[5/5] Testing endpoints..."

echo ""
echo "--- [GET] /health ---"
curl -s http://localhost:$PORT/health | python3 -m json.tool

echo ""
echo "--- [GET] /model/info ---"
curl -s http://localhost:$PORT/model/info | python3 -m json.tool

echo ""
echo "--- [POST] /predict with sample patient input ---"
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -X POST http://localhost:$PORT/predict \
    -H "Content-Type: application/json" \
    -d '{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }')

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | grep -v "HTTP_STATUS")

echo "HTTP Status: $HTTP_STATUS"
if [ "$HTTP_STATUS" = "200" ]; then
    echo "$BODY" | python3 -m json.tool
    echo ""
    echo "Prediction test PASSED ✓"
else
    echo "Prediction test FAILED. Response:"
    echo "$BODY"
    echo ""
    echo "Container logs:"
    docker logs $CONTAINER_NAME
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "Stopping container..."
docker rm -f $CONTAINER_NAME
echo "  Container stopped ✓"

echo ""
echo "============================================"
echo " Docker build and test complete ✓"
echo "============================================"
echo ""
echo "To run manually:"
echo "  docker run -p 8000:8000 \\"
echo "    -v \$(pwd)/src/models:/app/models \\"
echo "    -e MODEL_PATH=/app/models/best_model.pkl \\"
echo "    -e CONFIG_PATH=/app/models/feature_config.json \\"
echo "    $IMAGE_NAME"
