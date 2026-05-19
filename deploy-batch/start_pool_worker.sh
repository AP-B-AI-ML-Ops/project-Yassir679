#!/bin/bash
set -e

# Start the flow: registers the deployment with Prefect server and polls for scheduled runs
python /app/flow.py
