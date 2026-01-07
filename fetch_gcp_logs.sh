#!/bin/bash
gcloud logging read 'resource.type="cloud_run_revision" AND (textPayload=~"591003669450" OR textPayload=~"591002097996")' \
  --limit=100 \
  --format="table(timestamp,textPayload)" \
  --project=allmart-dashboard



