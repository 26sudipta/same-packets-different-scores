#!/bin/bash
# Phase 7 retrained label variants, CICIDS2017; resumable (cells checkpointed)
cd "$(dirname "$0")/../.."
.venv/bin/python 03_Code/phase7_retrain.py fold CICIDS2017 && \
.venv/bin/python 03_Code/phase7_retrain.py own CICIDS2017
