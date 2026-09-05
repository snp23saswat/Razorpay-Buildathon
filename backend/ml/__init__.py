"""
backend/ml/__init__.py
ML Registry and loader for the Revenue Recovery Agent
"""

import os
import sys
import joblib

ML_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

MODELS_DIR = os.path.join(ML_DIR, "models")

_REGISTRY = {}


def load_all_models():
    global _REGISTRY
    if _REGISTRY:
        return _REGISTRY

    # Ensure models exist; if not, train them
    sr_path = os.path.join(MODELS_DIR, "smart_retry_model.joblib")
    if not os.path.exists(sr_path):
        from .train_models import train_and_export
        train_and_export()

    try:
        _REGISTRY["smart_retry"] = joblib.load(os.path.join(MODELS_DIR, "smart_retry_model.joblib"))
    except Exception as e:
        print(f"Warning: could not load smart_retry model: {e}")

    try:
        _REGISTRY["intent_classifier"] = joblib.load(os.path.join(MODELS_DIR, "intent_classifier.joblib"))
    except Exception as e:
        print(f"Warning: could not load intent_classifier: {e}")

    try:
        _REGISTRY["uplift_optimizer"] = joblib.load(os.path.join(MODELS_DIR, "uplift_optimizer.joblib"))
    except Exception as e:
        print(f"Warning: could not load uplift_optimizer: {e}")

    try:
        _REGISTRY["invoice_risk"] = joblib.load(os.path.join(MODELS_DIR, "invoice_risk_model.joblib"))
    except Exception as e:
        print(f"Warning: could not load invoice_risk: {e}")

    return _REGISTRY


def get_model(name):
    reg = load_all_models()
    return reg.get(name)
