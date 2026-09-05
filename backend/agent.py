"""
agent.py — Enterprise Autonomous Revenue Recovery Agent (ML-Powered)

Architecture & 5-Stage Autonomous Loop:
  1. DETECT    -> Schema normalization, IST timezone coercion, customer profile binding.
  2. COMPLY    -> Unconditional consent gate (opt-out halts), daily customer contact deduplication,
                  and IST quiet-hours window enforcement (21:00–08:00 IST).
  3. DIAGNOSE  -> Hybrid intelligence:
                  - Deterministic root-cause mapping for structured gateway error codes
                  - ML Intent Classifier for ambiguous checkout drop-off sessions
                  - ML Risk Regressor for B2B invoice aging & default propensity
  4. DECIDE    -> ML-guided Policy Engine:
                  - Dynamic Discount & Net-Uplift Optimizer for checkouts
                  - Smart Retry Window & Propensity Optimizer for payment failures
                  - Bounded ladder: max 3 attempts, cooldown enforcement, human escalation limit
  5. AUDIT     -> Transparent, auditable trace with XAI feature rationales, IST timestamps,
                  exact cost deductions, and net revenue attribution.

Run: python agent.py > results.json
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta, timezone

# Ensure local backend/ml can be imported
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BACKEND_DIR, "ml")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from ml import get_model, load_all_models
from generate_data import generate_batch, IST

random.seed(42)

MAX_ATTEMPTS = 3
QUIET_HOURS = (21, 8)  # 21:00 (9pm) to 08:00 (8am) IST
MAX_DAILY_CONTACTS_PER_CUSTOMER = 1

# Base Intervention Cost Matrix (% of recovered amount or flat fee in INR)
INTERVENTION_COSTS = {
    "retry_after_delay": 5.0,            # Gateway retry fee
    "mandate_retry": 5.0,                # UPI Autopay / e-mandate retry fee
    "switch_payment_method_nudge": 10.0, # WhatsApp interactive payment link message
    "update_card_link": 10.0,            # Tokenized card update SMS/WhatsApp
    "update_payment_method_nudge": 10.0, # In-app / WhatsApp nudge
    "reminder_message": 8.0,             # Rich WhatsApp cart recovery notification
    "assist_message_with_faq": 12.0,     # Conversational support FAQ link
    "targeted_discount_nudge": 0.05,     # 5% baseline (overridden dynamically by ML uplift optimizer)
    "gentle_reminder": 15.0,             # Formal email & statement dispatch
    "promise_to_pay_followup": 25.0,     # Payment commitment tracking & automated SMS
    "formal_notice": 50.0,               # Registered formal dunning notification
    "human_escalation": 250.0,           # Dedicated recovery agent time cost
}


def parse_ist_datetime(ts_val):
    """Parses ISO string into IST-aware datetime."""
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=IST)
        return ts_val.astimezone(IST)
    try:
        dt = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=IST)
        return dt.astimezone(IST)
    except Exception:
        return datetime.now(IST)


def is_in_quiet_hours(dt):
    """Checks if IST time falls in quiet window (21:00 to 08:00 IST)."""
    hour = dt.hour
    start, end = QUIET_HOURS
    return hour >= start or hour < end


# ==========================================
# DIAGNOSIS & ML INFERENCE STAGE
# ==========================================

def diagnose_event(event, ml_models):
    """
    Hybrid Diagnosis:
    - Structured error codes -> deterministic rule + ML retry prediction
    - Checkout drop-offs -> ML Abandonment Intent Classifier
    - B2B Invoices -> ML Invoice Default Risk Regressor
    """
    etype = event["type"]
    signals = event["signals"]
    amount = event["amount"]

    if etype == "payment_failure" or etype == "subscription_failure":
        error_code = signals.get("gateway_error_code", "bank_technical_error")
        rule_conf = 0.95 if error_code != "bank_technical_error" else 0.75

        # Query ML Smart Retry Optimizer
        sr_model = ml_models.get("smart_retry")
        ml_retry_rec = None
        if sr_model is not None:
            features = {
                "gateway_error_code": error_code,
                "mandate_type": signals.get("mandate_type", "none"),
                "card_type": signals.get("card_type", "visa_credit"),
                "amount": amount,
                "attempt_number": signals.get("attempt_number", 1),
                "customer_tenure_months": signals.get("customer_tenure_months", 12),
                "hour_of_day": signals.get("hour_of_day", 14),
                "day_of_month": signals.get("day_of_month", 15),
            }
            try:
                ml_retry_rec = sr_model.recommend_optimal_retry_window(features)
            except Exception as e:
                pass

        return {
            "cause": error_code,
            "confidence": rule_conf,
            "method": "rule_assisted_by_ml_propensity",
            "ml_retry_rec": ml_retry_rec,
            "rationale": f"Mapped structured gateway signal '{error_code}' with high diagnostic confidence.",
        }

    elif etype == "checkout_abandonment":
        intent_model = ml_models.get("intent_classifier")
        uplift_model = ml_models.get("uplift_optimizer")

        import pandas as pd
        feat_df = pd.DataFrame([{
            "time_on_checkout_sec": signals.get("time_on_checkout_sec", 120),
            "cart_items": signals.get("cart_items", 2),
            "cart_amount": amount,
            "coupon_attempts": signals.get("coupon_attempts", 0),
            "scroll_depth_pct": signals.get("scroll_depth_pct", 75),
            "exit_velocity_px_sec": signals.get("exit_velocity_px_sec", 400),
            "past_purchases": signals.get("past_purchases", 1),
            "device": signals.get("device", "mobile"),
            "payment_method_selected": signals.get("payment_method_selected", "none"),
        }])

        intent_res = intent_model.predict_intent(feat_df) if intent_model else {
            "intent": "price_sensitivity" if signals.get("coupon_attempted") else "friction_hesitation",
            "confidence": 0.70,
            "top_factors": ["Heuristic classification"],
            "probabilities": {},
        }

        # Query ML Uplift Optimizer for dynamic discounting
        uplift_res = None
        if uplift_model is not None:
            uplift_dict = {
                "cart_amount": amount,
                "cart_items": signals.get("cart_items", 2),
                "time_on_checkout_sec": signals.get("time_on_checkout_sec", 120),
                "coupon_attempts": signals.get("coupon_attempts", 0),
                "past_purchases": signals.get("past_purchases", 1),
            }
            try:
                uplift_res = uplift_model.optimize_intervention(uplift_dict)
            except Exception:
                pass

        return {
            "cause": intent_res["intent"],
            "confidence": intent_res["confidence"],
            "method": "ml_gradient_boosting_classifier",
            "top_factors": intent_res.get("top_factors", []),
            "class_probabilities": intent_res.get("probabilities", {}),
            "ml_uplift_rec": uplift_res,
            "rationale": f"ML model detected intent '{intent_res['intent']}' with {intent_res['confidence']*100:.1f}% confidence based on session telemetry.",
        }

    elif etype == "overdue_invoice":
        inv_model = ml_models.get("invoice_risk")
        signals_dict = {
            "account_tier": signals.get("account_tier", "smb"),
            "has_dispute": str(signals.get("has_dispute", False)),
            "prior_promise_to_pay": str(signals.get("prior_promise_to_pay", False)),
            "amount": amount,
            "days_overdue": signals.get("days_overdue", 15),
            "invoice_count_open": signals.get("invoice_count_open", 1),
            "client_relationship_months": signals.get("client_relationship_months", 12),
        }

        inv_res = inv_model.assess_invoice(signals_dict) if inv_model else {
            "risk_tier": "medium_risk",
            "default_probability": 0.35,
            "confidence": 0.8,
            "recommended_strategy": "structured_payment_plan",
            "action_ladder": [("gentle_reminder", 0.6), ("formal_notice", 0.3), ("human_escalation", 0.2)],
        }

        return {
            "cause": inv_res["risk_tier"],
            "confidence": inv_res["confidence"],
            "method": "ml_random_forest_risk_model",
            "default_probability": inv_res["default_probability"],
            "expected_additional_days": inv_res.get("expected_additional_days", 10),
            "recommended_strategy": inv_res["recommended_strategy"],
            "recommended_tone": inv_res.get("recommended_tone", "Professional Notice"),
            "custom_ladder": inv_res.get("action_ladder"),
            "rationale": f"B2B Risk Model classified as '{inv_res['risk_tier']}' (Default prob: {inv_res['default_probability']*100:.1f}%).",
        }

    return {
        "cause": "unknown_anomaly",
        "confidence": 0.4,
        "method": "fallback_rule",
        "rationale": "Unrecognized event type, defaulting to safe ladder.",
    }


# ==========================================
# ML-GUIDED POLICY & INTERVENTION LADDER
# ==========================================

def build_intervention_ladder(event, diagnosis):
    """
    Constructs an optimal, bounded intervention ladder guided by ML predictions.
    """
    etype = event["type"]
    cause = diagnosis["cause"]

    if etype == "payment_failure":
        rec = diagnosis.get("ml_retry_rec")
        best_p = rec["best_prob"] if rec else 0.60
        if cause == "insufficient_funds":
            return [
                ("retry_after_delay", best_p, {"retry_window": rec.get("recommended_time_slot") if rec else "+12h"}),
                ("switch_payment_method_nudge", 0.45, {"channel": "whatsapp_upi_intent"}),
                ("human_escalation", 0.20, {}),
            ]
        elif cause == "expired_card":
            return [
                ("update_card_link", 0.72, {"channel": "tokenized_sms_link"}),
                ("switch_payment_method_nudge", 0.38, {}),
                ("human_escalation", 0.15, {}),
            ]
        elif cause == "bank_technical_error":
            return [
                ("retry_after_delay", best_p, {"retry_window": rec.get("recommended_time_slot") if rec else "+2h"}),
                ("retry_after_delay", max(0.4, best_p - 0.2), {"retry_window": "+8h"}),
                ("human_escalation", 0.25, {}),
            ]
        elif cause == "risk_blocked":
            return [("human_escalation", 0.35, {"reason": "fraud_prevention_review"})]
        else:
            return [
                ("retry_after_delay", best_p, {}),
                ("switch_payment_method_nudge", 0.40, {}),
                ("human_escalation", 0.20, {}),
            ]

    elif etype == "subscription_failure":
        rec = diagnosis.get("ml_retry_rec")
        best_p = rec["best_prob"] if rec else 0.55
        return [
            ("mandate_retry", best_p, {"retry_window": rec.get("recommended_time_slot") if rec else "+24h"}),
            ("update_payment_method_nudge", 0.45, {"channel": "in_app_mandate_update"}),
            ("human_escalation", 0.18, {}),
        ]

    elif etype == "checkout_abandonment":
        uplift = diagnosis.get("ml_uplift_rec")
        if uplift:
            best_action = uplift["action_name"]
            discount_pct = uplift["recommended_discount_pct"]
            expected_p = uplift["tier_analysis"][discount_pct // 5]["predicted_conversion_prob"] if discount_pct <= 15 else 0.45

            if best_action == "targeted_discount_nudge":
                return [
                    ("targeted_discount_nudge", expected_p, {"discount_pct": discount_pct, "coupon_code": f"SAVE{discount_pct}NOW"}),
                    ("reminder_message", 0.25, {"discount_pct": 0}),
                    ("human_escalation", 0.08, {}),
                ]
            else:
                return [
                    ("reminder_message", expected_p, {"discount_pct": 0, "urgency": "low_stock"}),
                    ("assist_message_with_faq", 0.32, {}),
                    ("human_escalation", 0.05, {}),
                ]

        if cause == "price_sensitivity":
            return [
                ("targeted_discount_nudge", 0.52, {"discount_pct": 10, "coupon_code": "SAVE10NOW"}),
                ("reminder_message", 0.28, {}),
                ("human_escalation", 0.05, {}),
            ]
        elif cause == "friction_hesitation":
            return [
                ("assist_message_with_faq", 0.42, {"include_faq": True}),
                ("reminder_message", 0.30, {}),
            ]
        elif cause == "technical_issue":
            return [
                ("switch_payment_method_nudge", 0.48, {"message": "Alternative one-click UPI checkout"}),
                ("assist_message_with_faq", 0.25, {}),
            ]
        else:
            return [("reminder_message", 0.32, {"highlight_items": True})]

    elif etype == "overdue_invoice":
        custom_ladder = diagnosis.get("custom_ladder")
        if custom_ladder:
            return [(act, prob, {"strategy": diagnosis.get("recommended_strategy")}) for act, prob in custom_ladder]
        return [
            ("gentle_reminder", 0.60, {}),
            ("formal_notice", 0.35, {}),
            ("human_escalation", 0.20, {}),
        ]

    return [("human_escalation", 0.25, {})]


# ==========================================
# CORE AGENT ENGINE
# ==========================================

def run_recovery_agent(events, seed=42):
    random.seed(seed)
    ml_models = load_all_models()

    audit_trail = []
    summary = {
        "total_at_risk": 0.0,
        "total_recovered": 0.0,
        "total_intervention_cost": 0.0,
        "net_value_recovered": 0.0,
        "roi_multiplier": 0.0,
        "recovery_rate": 0.0,
        "by_type": {},
        "stopped_for_optout": 0,
        "quiet_hours_deferred": 0,
        "customer_dedup_capped": 0,
        "escalated_to_human": 0,
        "recovered_count": 0,
        "not_recovered": 0,
        "processed_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }

    # Tracking per-customer contact count to enforce daily spam prevention
    customer_daily_contacts = {}

    for event in events:
        etype = event.get("type", "payment_failure")
        amount = float(event.get("amount", 1000.0))
        cust = event.get("customer", {"id": "cust_unknown", "name": "Customer", "opted_out": False})
        cid = cust.get("id", "cust_unknown")
        event_ts = parse_ist_datetime(event.get("timestamp", datetime.now(IST).isoformat()))

        summary["total_at_risk"] += amount
        summary["by_type"].setdefault(etype, {
            "at_risk": 0.0,
            "recovered": 0.0,
            "cost": 0.0,
            "net": 0.0,
            "count": 0,
            "recovered_count": 0,
        })
        summary["by_type"][etype]["at_risk"] += amount
        summary["by_type"][etype]["count"] += 1

        log = {
            "event_id": event["event_id"],
            "type": etype,
            "customer_id": cid,
            "customer_name": cust.get("name", "Unknown"),
            "customer_lang": cust.get("lang", "en"),
            "amount": amount,
            "timestamp_ist": event_ts.strftime("%Y-%m-%d %H:%M:%S IST"),
            "steps": [],
            "outcome": None,
            "recovered_amount": 0.0,
            "total_cost": 0.0,
            "net_recovered": 0.0,
            "ml_diagnostics": {},
        }

        # -------------------------------------------------------------
        # STAGE 1: COMPLIANCE & CONSENT GATE (Unconditional Opt-out Check)
        # -------------------------------------------------------------
        if cust.get("opted_out", False):
            log["steps"].append({
                "stage": "compliance_gate",
                "action": "check_consent",
                "result": "BLOCKED — Customer explicitly opted out. Mandatory compliance halt; no contact permitted.",
                "cost": 0.0,
            })
            log["outcome"] = "stopped_no_contact"
            summary["stopped_for_optout"] += 1
            audit_trail.append(log)
            continue

        # -------------------------------------------------------------
        # STAGE 2: SAME-DAY CUSTOMER CONTACT DEDUPLICATION GATE
        # -------------------------------------------------------------
        date_key = (cid, event_ts.strftime("%Y-%m-%d"))
        current_contacts = customer_daily_contacts.get(date_key, 0)
        if current_contacts >= MAX_DAILY_CONTACTS_PER_CUSTOMER:
            log["steps"].append({
                "stage": "compliance_gate",
                "action": "contact_frequency_cap",
                "result": f"DEFERRED — Customer {cid} already contacted on {date_key[1]} ({MAX_DAILY_CONTACTS_PER_CUSTOMER}/day limit). Queued for next day window.",
                "cost": 0.0,
            })
            summary["customer_dedup_capped"] += 1
            log["outcome"] = "deferred_frequency_cap"
            audit_trail.append(log)
            continue

        # -------------------------------------------------------------
        # STAGE 3: HYBRID ML & RULE DIAGNOSIS
        # -------------------------------------------------------------
        diagnosis = diagnose_event(event, ml_models)
        log["ml_diagnostics"] = {
            "diagnosed_cause": diagnosis["cause"],
            "confidence_score": diagnosis["confidence"],
            "method": diagnosis["method"],
            "rationale": diagnosis["rationale"],
        }
        if "ml_retry_rec" in diagnosis and diagnosis["ml_retry_rec"]:
            log["ml_diagnostics"]["smart_retry_recommendation"] = diagnosis["ml_retry_rec"]["recommended_time_slot"]
        if "ml_uplift_rec" in diagnosis and diagnosis["ml_uplift_rec"]:
            log["ml_diagnostics"]["uplift_optimal_discount"] = f"{diagnosis['ml_uplift_rec']['recommended_discount_pct']}%"

        log["steps"].append({
            "stage": "diagnosis",
            "action": "diagnose_root_cause",
            "cause": diagnosis["cause"],
            "confidence": diagnosis["confidence"],
            "method": diagnosis["method"],
            "result": diagnosis["rationale"],
            "cost": 0.0,
        })

        # -------------------------------------------------------------
        # STAGE 4: ML-GUIDED POLICY & BOUNDED INTERVENTION LADDER
        # -------------------------------------------------------------
        ladder = build_intervention_ladder(event, diagnosis)
        recovered = False
        event_cost = 0.0

        for attempt_idx, item in enumerate(ladder[:MAX_ATTEMPTS]):
            action = item[0]
            base_prob = item[1]
            meta = item[2] if len(item) > 2 else {}

            attempt_ts = event_ts + timedelta(hours=12 * attempt_idx)

            # IST Quiet Hours Check (21:00 - 08:00 IST)
            if action != "human_escalation" and is_in_quiet_hours(attempt_ts):
                log["steps"].append({
                    "stage": "policy_enforcement",
                    "action": action,
                    "attempt": attempt_idx + 1,
                    "result": f"DEFERRED — Inside quiet hours ({attempt_ts.strftime('%H:%M')} IST). Scheduled for 08:30 IST window.",
                    "cost": 0.0,
                })
                summary["quiet_hours_deferred"] += 1
                # Shift execution to next morning 08:30 IST
                attempt_ts = attempt_ts.replace(hour=8, minute=30)

            # Human Escalation Stop
            if action == "human_escalation":
                cost = INTERVENTION_COSTS.get("human_escalation", 250.0)
                event_cost += cost
                log["steps"].append({
                    "stage": "escalation",
                    "action": "human_escalation",
                    "attempt": attempt_idx + 1,
                    "result": "Escalated to specialized human recovery team (automated ladder exhausted).",
                    "cost": cost,
                })
                summary["escalated_to_human"] += 1
                break

            # Execute intervention
            customer_daily_contacts[date_key] = customer_daily_contacts.get(date_key, 0) + 1
            success = random.random() < base_prob

            # Calculate precise cost
            if action == "targeted_discount_nudge":
                disc_rate = (meta.get("discount_pct", 5)) / 100.0
                cost = (amount * disc_rate) + 5.0  # discount cost + messaging fee
            else:
                cost = INTERVENTION_COSTS.get(action, 10.0)

            event_cost += cost

            step_detail = f"Executed {action} via {meta.get('channel', 'automated_engine')}."
            if "retry_window" in meta:
                step_detail += f" (Smart Window: {meta['retry_window']})"
            if "discount_pct" in meta:
                step_detail += f" (ML Discount: {meta['discount_pct']}%)"

            log["steps"].append({
                "stage": "execution",
                "action": action,
                "attempt": attempt_idx + 1,
                "success_probability": round(base_prob, 3),
                "result": "RECOVERED — Payment confirmed" if success else "FAILED — Customer did not complete transaction",
                "details": step_detail,
                "cost": round(cost, 2),
            })

            if success:
                recovered = True
                break

        # -------------------------------------------------------------
        # STAGE 5: AUDIT ROLLUP & FINANCIAL METRICS
        # -------------------------------------------------------------
        log["total_cost"] = round(event_cost, 2)
        summary["total_intervention_cost"] += event_cost
        summary["by_type"][etype]["cost"] += event_cost

        if recovered:
            net = amount - event_cost
            log["outcome"] = "recovered"
            log["recovered_amount"] = round(amount, 2)
            log["net_recovered"] = round(net, 2)

            summary["total_recovered"] += amount
            summary["by_type"][etype]["recovered"] += amount
            summary["by_type"][etype]["recovered_count"] += 1
            summary["recovered_count"] += 1
        else:
            log["outcome"] = "not_recovered"
            log["net_recovered"] = round(-event_cost, 2)
            summary["not_recovered"] += 1

        audit_trail.append(log)

    # Summary calculations
    summary["total_at_risk"] = round(summary["total_at_risk"], 2)
    summary["total_recovered"] = round(summary["total_recovered"], 2)
    summary["total_intervention_cost"] = round(summary["total_intervention_cost"], 2)
    summary["net_value_recovered"] = round(summary["total_recovered"] - summary["total_intervention_cost"], 2)
    
    summary["recovery_rate"] = round(summary["total_recovered"] / summary["total_at_risk"], 4) if summary["total_at_risk"] > 0 else 0.0
    summary["roi_multiplier"] = round(summary["net_value_recovered"] / summary["total_intervention_cost"], 2) if summary["total_intervention_cost"] > 0 else 0.0

    for v in summary["by_type"].values():
        v["at_risk"] = round(v["at_risk"], 2)
        v["recovered"] = round(v["recovered"], 2)
        v["cost"] = round(v["cost"], 2)
        v["net"] = round(v["recovered"] - v["cost"], 2)
        v["recovery_rate"] = round(v["recovered"] / v["at_risk"], 4) if v["at_risk"] > 0 else 0.0

    return {"summary": summary, "audit_trail": audit_trail}


if __name__ == "__main__":
    events = generate_batch(n_per_type=15, seed=42)
    result = run_recovery_agent(events, seed=42)
    
    # Write to backend results.json
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also copy to dashboard/data.json so static dashboard has fresh data
    dashboard_data_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "data.json")
    if os.path.exists(os.path.dirname(dashboard_data_path)):
        with open(dashboard_data_path, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
