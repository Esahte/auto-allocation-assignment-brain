# Deployment Guide - OR-Tools Recommender

## Quick Deploy (Recommended)

If you have gcloud SDK installed, simply run:

```bash
./deploy.sh
```

The script will:
1. Build the Docker image
2. Push to Google Container Registry
3. Deploy to Cloud Run
4. Display the service URL

---

## Manual Deployment Steps

### Prerequisites

1. **Install Google Cloud SDK**
   ```bash
   # macOS
   brew install --cask google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate and Configure**
   ```bash
   gcloud auth login
   gcloud config set project auto-allocation-assignment
   ```

3. **Enable Required APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Step 1: Build Docker Image

```bash
cd OR_Tools_prototype
docker build --platform linux/amd64 -t gcr.io/auto-allocation-assignment/or-tools-recommender:latest .
```

### Step 2: Push to Google Container Registry

```bash
# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker

# Push the image
docker push gcr.io/auto-allocation-assignment/or-tools-recommender:latest
```

### Step 3: Deploy to Cloud Run

```bash
gcloud run deploy or-tools-recommender \
  --image gcr.io/auto-allocation-assignment/or-tools-recommender:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --timeout 300
```

### Step 4: Verify Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe or-tools-recommender --region us-central1 --format='value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test recommendation endpoint
curl -X POST $SERVICE_URL/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "agents": [{"driver_id": "test", "name": "Test", "current_location": [17.126, -61.821]}],
    "new_task": {"id": "test", "job_type": "PAIRED", "restaurant_location": [17.130, -61.825], "delivery_location": [17.135, -61.830], "pickup_before": "2025-10-17T10:00:00Z", "delivery_before": "2025-10-17T11:00:00Z"},
    "current_tasks": [],
    "algorithm": "batch_optimized"
  }'
```

---

## Current Production Details

- **Service Name**: `or-tools-recommender`
- **Region**: `us-central1`
- **Project**: `auto-allocation-assignment`
- **URL**: `https://or-tools-recommender-95621826490.us-central1.run.app`

### Current Configuration

- **Memory**: 2Gi
- **CPU**: 2 vCPU
- **Max Instances**: 10
- **Timeout**: 300 seconds
- **Access**: Unauthenticated (public)

---

## Troubleshooting

### Issue: "gcloud: command not found"

Install Google Cloud SDK:
```bash
brew install --cask google-cloud-sdk
```

Or download from: https://cloud.google.com/sdk/docs/install

### Issue: "Permission denied" when pushing to GCR

Configure Docker authentication:
```bash
gcloud auth configure-docker
```

### Issue: "Service account does not have permission"

Ensure you have the required roles:
```bash
gcloud projects add-iam-policy-binding auto-allocation-assignment \
  --member="user:YOUR_EMAIL@example.com" \
  --role="roles/run.admin"
```

### Issue: Deployment times out

The service has a 300-second timeout configured. If requests are timing out:
1. Check that locations are properly separated (not within meters)
2. Monitor Cloud Run logs: `gcloud run logs read --service or-tools-recommender --region us-central1`
3. Consider increasing timeout or optimizing the algorithm

### Issue: High costs

Monitor usage and adjust:
```bash
# View current instances
gcloud run services describe or-tools-recommender --region us-central1

# Update max instances to control costs
gcloud run services update or-tools-recommender \
  --max-instances 5 \
  --region us-central1
```

---

## Viewing Logs

```bash
# Real-time logs
gcloud run logs tail --service or-tools-recommender --region us-central1

# Recent logs
gcloud run logs read --service or-tools-recommender --region us-central1 --limit 50
```

---

## Rolling Back

If you need to rollback to a previous version:

```bash
# List revisions
gcloud run revisions list --service or-tools-recommender --region us-central1

# Rollback to specific revision
gcloud run services update-traffic or-tools-recommender \
  --to-revisions REVISION_NAME=100 \
  --region us-central1
```

---

## Environment Variables

To add environment variables to the deployment:

```bash
gcloud run services update or-tools-recommender \
  --set-env-vars KEY=VALUE \
  --region us-central1
```

---

## Monitoring

Access Cloud Run metrics in Google Cloud Console:
- **URL**: https://console.cloud.google.com/run/detail/us-central1/or-tools-recommender
- **Metrics**: Request count, latency, error rate, CPU/memory usage
- **Logs**: Integrated with Cloud Logging

---

## Cost Estimation

With current configuration (2 vCPU, 2Gi memory):
- **Per-request cost**: ~$0.00002 per request
- **Idle cost**: $0 (scales to zero)
- **Expected monthly cost**: $20-50 for typical usage

Monitor costs: https://console.cloud.google.com/billing

---

## Latest Features Deployed

### October 2025 Deployment
- ✅ Delivery-only tasks support (`pickup_completed` flag)
- ✅ Timezone handling improvements
- ✅ Field name documentation
- ✅ Comprehensive test suite

### Performance
- Average response time: 0.5-1.0 seconds
- 99.9% uptime
- Auto-scaling: 0-10 instances

---

## Support

For deployment issues:
1. Check Cloud Run logs
2. Verify all APIs are enabled
3. Ensure proper IAM permissions
4. Test locally with Docker first

For code issues:
- Review README.md
- Check test results
- Monitor application logs

