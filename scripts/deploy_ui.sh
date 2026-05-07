#!/bin/bash
# =============================================================================
# scripts/deploy_ui.sh
# Deploys the HTML prediction UI to GKE using ConfigMap + nginx
# Usage: bash scripts/deploy_ui.sh
# =============================================================================

set -e

NAMESPACE="mlops"
UI_FILE="ui/index.html"
API_IP="34.31.48.169"

echo "============================================"
echo " Heart Disease UI - GKE Deployment"
echo "============================================"

# ── Step 1: Check UI file exists ──────────────────────────────────────────────
echo ""
echo "[1/5] Checking UI file..."
if [ ! -f "$UI_FILE" ]; then
    echo "ERROR: $UI_FILE not found. Make sure ui/index.html exists."
    exit 1
fi
echo "  ui/index.html found ✓"

# ── Step 2: Create namespace if not exists ────────────────────────────────────
echo ""
echo "[2/5] Ensuring namespace exists..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
echo "  Namespace: $NAMESPACE ✓"

# ── Step 3: Create ConfigMap from HTML file ───────────────────────────────────
echo ""
echo "[3/5] Creating ConfigMap from ui/index.html..."
kubectl create configmap heart-disease-ui-content \
    --from-file=index.html=$UI_FILE \
    --namespace=$NAMESPACE \
    --dry-run=client -o yaml | kubectl apply -f -
echo "  ConfigMap created ✓"

# ── Step 4: Deploy nginx + service ───────────────────────────────────────────
echo ""
echo "[4/5] Deploying nginx + LoadBalancer service..."
cat << YAML | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-disease-ui
  namespace: $NAMESPACE
  labels:
    app: heart-disease-ui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: heart-disease-ui
  template:
    metadata:
      labels:
        app: heart-disease-ui
    spec:
      containers:
        - name: heart-disease-ui
          image: nginx:alpine
          ports:
            - containerPort: 80
          volumeMounts:
            - name: ui-content
              mountPath: /usr/share/nginx/html
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "100m"
              memory: "128Mi"
      volumes:
        - name: ui-content
          configMap:
            name: heart-disease-ui-content
---
apiVersion: v1
kind: Service
metadata:
  name: heart-disease-ui-service
  namespace: $NAMESPACE
  labels:
    app: heart-disease-ui
spec:
  type: LoadBalancer
  selector:
    app: heart-disease-ui
  ports:
    - port: 80
      targetPort: 80
      protocol: TCP
YAML
echo "  Deployment and Service applied ✓"

# ── Step 5: Wait for rollout and get IP ──────────────────────────────────────
echo ""
echo "[5/5] Waiting for deployment to be ready..."
kubectl rollout status deployment/heart-disease-ui -n $NAMESPACE --timeout=120s
echo "  Deployment ready ✓"

echo ""
echo "Waiting for External IP (this may take 1-2 minutes)..."
for i in $(seq 1 20); do
    UI_IP=$(kubectl get service heart-disease-ui-service -n $NAMESPACE \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -n "$UI_IP" ]; then
        break
    fi
    echo "  Waiting... ($i/20)"
    sleep 10
done

echo ""
echo "============================================"
echo " UI Deployment Complete!"
echo "============================================"
kubectl get service heart-disease-ui-service -n $NAMESPACE
echo ""
if [ -n "$UI_IP" ]; then
    echo "Open in browser: http://$UI_IP"
    echo ""
    echo "The API URL is pre-configured to: http://$API_IP:8000"
    echo "Just click 'Predict Risk' to test!"
else
    echo "External IP still pending. Run this to check:"
    echo "  kubectl get service heart-disease-ui-service -n mlops"
fi
