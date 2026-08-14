#!/bin/bash
# Usage: ./fetch_gcp_logs.sh <search-term> [limit] [service]
#   ./fetch_gcp_logs.sh 2159161
#   ./fetch_gcp_logs.sh 2159161 200 or-tools-recommender
#   ./fetch_gcp_logs.sh 620151386008717857053993800030 200 allmart-dashboard
set -euo pipefail

SEARCH="${1:?usage: $0 <search-term> [limit] [service]}"
LIMIT="${2:-100}"
SERVICE="${3:-}"

FILTER="resource.type=\"cloud_run_revision\" AND textPayload:\"${SEARCH}\""
if [ -n "$SERVICE" ]; then
  FILTER="${FILTER} AND resource.labels.service_name=\"${SERVICE}\""
fi

gcloud logging read "$FILTER" \
  --limit="${LIMIT}" \
  --format="table(timestamp,textPayload)" \
  --project=resonant-tube-437220-t3
