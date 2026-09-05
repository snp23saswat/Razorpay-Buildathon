"""
train_models.py — Fast Training & Serialization Pipeline for ML Models
"""

import os
import sys
import json
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_smart_retry import SmartRetryOptimizer
from ml_intent_classifier import AbandonmentIntentClassifier
from ml_uplift_optimizer import UpliftOptimizer
from ml_invoice_risk import InvoiceRiskModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_smart_retry_data(n=600):
    np.random.seed(42)
    random.seed(42)
    error_codes = ["insufficient_funds", "card_declined_generic", "bank_technical_error", "expired_card", "risk_blocked"]
    mandate_types = ["upi_autopay", "card_emandate", "netbanking", "none"]
    card_types = ["visa_credit", "mastercard_debit", "rupay_debit", "amex_credit"]
    
    rows = []
    y = []
    for _ in range(n):
        code = random.choice(error_codes)
        mandate = random.choice(mandate_types)
        card = random.choice(card_types)
        amount = float(np.random.exponential(scale=3500) + 199)
        attempt = random.randint(1, 4)
        tenure = random.randint(1, 36)
        hour = random.randint(0, 23)
        dom = random.randint(1, 31)
        
        base_p = 0.55
        if code == "bank_technical_error":
            base_p = 0.78
        elif code == "insufficient_funds":
            is_payday = (dom <= 7 or dom >= 28)
            base_p = 0.65 if is_payday else 0.42
        elif code == "expired_card":
            base_p = 0.25
        elif code == "risk_blocked":
            base_p = 0.12
            
        if 9 <= hour <= 12 or 17 <= hour <= 20:
            base_p += 0.12
        elif hour >= 22 or hour < 7:
            base_p -= 0.20
            
        base_p -= (attempt - 1) * 0.15
        base_p = float(np.clip(base_p, 0.05, 0.95))
        recovered = 1 if random.random() < base_p else 0
        
        rows.append({
            "gateway_error_code": code,
            "mandate_type": mandate,
            "card_type": card,
            "amount": round(amount, 2),
            "attempt_number": attempt,
            "customer_tenure_months": tenure,
            "hour_of_day": hour,
            "day_of_month": dom,
        })
        y.append(recovered)
    return pd.DataFrame(rows), np.array(y)


def generate_intent_data(n=600):
    np.random.seed(42)
    random.seed(42)
    devices = ["mobile", "desktop", "tablet"]
    payment_methods = ["upi", "card", "netbanking", "cod", "none"]
    
    rows = []
    y = []
    for _ in range(n):
        cluster = random.choices(
            ["price_sensitivity", "friction_hesitation", "technical_issue", "distraction_or_bounce"],
            weights=[0.35, 0.25, 0.20, 0.20]
        )[0]
        device = random.choice(devices)
        pm = random.choice(payment_methods)
        past_purchases = random.randint(0, 15)
        
        if cluster == "price_sensitivity":
            coupon_attempts = random.randint(1, 5)
            time_on_sec = random.randint(70, 350)
            cart_items = random.randint(1, 6)
            cart_amount = float(np.random.uniform(1500, 12000))
            scroll_depth = random.randint(60, 100)
            exit_velocity = random.randint(200, 800)
        elif cluster == "friction_hesitation":
            coupon_attempts = 0
            time_on_sec = random.randint(120, 600)
            cart_items = random.randint(2, 8)
            cart_amount = float(np.random.uniform(800, 6000))
            scroll_depth = random.randint(80, 100)
            exit_velocity = random.randint(100, 500)
        elif cluster == "technical_issue":
            coupon_attempts = random.choice([0, 1])
            time_on_sec = random.randint(40, 200)
            cart_items = random.randint(1, 4)
            cart_amount = float(np.random.uniform(500, 5000))
            scroll_depth = random.randint(30, 80)
            exit_velocity = random.randint(600, 1500)
        else:
            coupon_attempts = 0
            time_on_sec = random.randint(5, 35)
            cart_items = 1
            cart_amount = float(np.random.uniform(299, 2000))
            scroll_depth = random.randint(10, 35)
            exit_velocity = random.randint(800, 2000)
            
        rows.append({
            "time_on_checkout_sec": time_on_sec,
            "cart_items": cart_items,
            "cart_amount": round(cart_amount, 2),
            "coupon_attempts": coupon_attempts,
            "scroll_depth_pct": scroll_depth,
            "exit_velocity_px_sec": exit_velocity,
            "past_purchases": past_purchases,
            "device": device,
            "payment_method_selected": pm,
        })
        y.append(cluster)
    return pd.DataFrame(rows), np.array(y)


def generate_uplift_data(n=600):
    np.random.seed(42)
    random.seed(42)
    rows = []
    y_conv = []
    for _ in range(n):
        cart_amount = float(np.random.uniform(399, 15000))
        cart_items = random.randint(1, 6)
        time_on_sec = random.randint(20, 450)
        coupon_attempts = random.randint(0, 4)
        past_purchases = random.randint(0, 12)
        discount = random.choice([0.0, 0.05, 0.10, 0.15])
        
        base_conv = 0.22
        if past_purchases > 2:
            base_conv += 0.15
        elasticity = 2.4 if coupon_attempts > 0 else 1.2
        prob = np.clip(base_conv + (discount * elasticity) - (cart_amount / 40000.0), 0.05, 0.92)
        
        rows.append({
            "cart_amount": round(cart_amount, 2),
            "cart_items": cart_items,
            "time_on_checkout_sec": time_on_sec,
            "coupon_attempts": coupon_attempts,
            "past_purchases": past_purchases,
            "discount_offered": discount,
        })
        y_conv.append(prob)
    return pd.DataFrame(rows), np.array(y_conv)


def generate_invoice_risk_data(n=600):
    np.random.seed(42)
    random.seed(42)
    tiers = ["smb", "mid_market", "enterprise"]
    rows = []
    y_class = []
    y_delay = []
    for _ in range(n):
        tier = random.choice(tiers)
        has_dispute = random.choice([True, False])
        promise = random.choice([True, False])
        amount = float(np.random.uniform(10000, 500000))
        days = random.randint(1, 75)
        open_invoices = random.randint(1, 5)
        rel_months = random.randint(2, 60)
        
        risk_score = (days / 80.0) * 0.5 + (0.3 if has_dispute else 0.0) + (0.15 if not promise else -0.1)
        if tier == "smb":
            risk_score += 0.15
        elif tier == "enterprise":
            risk_score -= 0.15
        risk_score = float(np.clip(risk_score, 0.05, 0.95))
        
        if risk_score > 0.6:
            risk_tier = "high_default_risk"
            delay = days + random.randint(20, 60)
        elif risk_score > 0.3:
            risk_tier = "medium_risk"
            delay = days + random.randint(7, 25)
        else:
            risk_tier = "low_risk"
            delay = days + random.randint(1, 10)
            
        rows.append({
            "account_tier": tier,
            "has_dispute": str(has_dispute),
            "prior_promise_to_pay": str(promise),
            "amount": round(amount, 2),
            "days_overdue": days,
            "invoice_count_open": open_invoices,
            "client_relationship_months": rel_months,
        })
        y_class.append(risk_tier)
        y_delay.append(delay)
    return pd.DataFrame(rows), np.array(y_class), np.array(y_delay)


def train_and_export():
    print("Training ML Models...", flush=True)
    metrics_summary = {}

    # 1. Smart Retry
    X_sr, y_sr = generate_smart_retry_data(600)
    split_sr = int(len(X_sr) * 0.8)
    sr_model = SmartRetryOptimizer()
    sr_model.train(X_sr.iloc[:split_sr], y_sr[:split_sr])
    sr_preds = sr_model.pipeline.predict(X_sr.iloc[split_sr:])
    sr_probs = sr_model.pipeline.predict_proba(X_sr.iloc[split_sr:])[:, 1]
    sr_acc = accuracy_score(y_sr[split_sr:], sr_preds)
    sr_auc = roc_auc_score(y_sr[split_sr:], sr_probs)
    joblib.dump(sr_model, os.path.join(MODELS_DIR, "smart_retry_model.joblib"))
    metrics_summary["smart_retry"] = {
        "model_type": "Random Forest Classifier",
        "accuracy": round(float(sr_acc), 4),
        "roc_auc": round(float(sr_auc), 4),
        "feature_importances": sr_model.get_feature_importances(),
        "status": "active_trained",
    }
    print(f"Smart Retry: Acc={sr_acc*100:.1f}%, AUC={sr_auc:.3f}", flush=True)

    # 2. Intent Classifier
    X_int, y_int = generate_intent_data(600)
    split_int = int(len(X_int) * 0.8)
    intent_model = AbandonmentIntentClassifier()
    intent_model.train(X_int.iloc[:split_int], y_int[:split_int])
    int_preds = intent_model.pipeline.predict(X_int.iloc[split_int:])
    int_acc = accuracy_score(y_int[split_int:], int_preds)
    joblib.dump(intent_model, os.path.join(MODELS_DIR, "intent_classifier.joblib"))
    metrics_summary["intent_classifier"] = {
        "model_type": "Gradient Boosting Classifier",
        "accuracy": round(float(int_acc), 4),
        "classes": intent_model.CLASSES,
        "feature_importances": intent_model.get_feature_importances(),
        "status": "active_trained",
    }
    print(f"Intent Classifier: Acc={int_acc*100:.1f}%", flush=True)

    # 3. Uplift Optimizer
    X_up, y_up = generate_uplift_data(600)
    split_up = int(len(X_up) * 0.8)
    uplift_model = UpliftOptimizer()
    uplift_model.train(X_up.iloc[:split_up], y_up[:split_up])
    up_preds = uplift_model.pipeline.predict(X_up.iloc[split_up:])
    up_r2 = r2_score(y_up[split_up:], up_preds)
    joblib.dump(uplift_model, os.path.join(MODELS_DIR, "uplift_optimizer.joblib"))
    metrics_summary["uplift_optimizer"] = {
        "model_type": "Random Forest Regressor",
        "r2_score": round(float(up_r2), 4),
        "candidate_discounts": [0, 5, 10, 15],
        "status": "active_trained",
    }
    print(f"Uplift Optimizer: R2={up_r2:.3f}", flush=True)

    # 4. Invoice Risk
    X_inv, y_inv_cls, y_inv_del = generate_invoice_risk_data(600)
    split_inv = int(len(X_inv) * 0.8)
    inv_model = InvoiceRiskModel()
    inv_model.train(X_inv.iloc[:split_inv], y_inv_cls[:split_inv], y_inv_del[:split_inv])
    inv_preds = inv_model.classifier_pipeline.predict(X_inv.iloc[split_inv:])
    inv_acc = accuracy_score(y_inv_cls[split_inv:], inv_preds)
    joblib.dump(inv_model, os.path.join(MODELS_DIR, "invoice_risk_model.joblib"))
    metrics_summary["invoice_risk"] = {
        "model_type": "Dual Classifier & Regressor",
        "risk_accuracy": round(float(inv_acc), 4),
        "risk_tiers": inv_model.RISK_TIERS,
        "status": "active_trained",
    }
    print(f"Invoice Risk: Acc={inv_acc*100:.1f}%", flush=True)

    with open(os.path.join(MODELS_DIR, "ml_insights.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Training completed successfully!", flush=True)
    return metrics_summary


if __name__ == "__main__":
    train_and_export()
