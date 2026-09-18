#!/bin/bash
# Runs the project. Use "./run.sh retrain" to rebuild the data table and model first.

cd "$(dirname "$0")"

# Turn on the virtual environment if one exists in this folder
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

if [ "$1" == "retrain" ] || [ ! -f model.joblib ]; then
    python step1_prepare_data.py || exit 1
    python step2_train_model.py || exit 1
fi

streamlit run step3_app.py
