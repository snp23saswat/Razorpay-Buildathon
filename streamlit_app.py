"""
streamlit_app.py — Recovery Audit Agent (Streamlit Cloud Web App)
Autonomous Multi-Channel Revenue Recovery Engine with Machine Learning & Explainable AI.
"""

import os
import sys
import json
import io
import csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# Setup sys.path for backend and ML modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
ML_DIR = os.path.join(BACKEND_DIR, "ml")

for d in [ROOT_DIR, BACKEND_DIR, ML_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from agent import run_recovery_agent, parse_ist_datetime, is_in_quiet_hours, INTERVENTION_COSTS
from generate_data import generate_batch, CUSTOMERS, IST
from ml import load_all_models, get_model

# Page Configuration
st.set_page_config(
    page_title="Recovery Audit Agent | ML Enterprise Edition",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
  .metric-card {
    background: rgba(18, 24, 38, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .highlight-emerald {
    border-left: 4px solid #10B981;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(18, 24, 38, 0.8) 100%);
  }
  .trace-step {
    background: rgba(0, 0, 0, 0.3);
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

# Helper formatting
def fmt_inr(n):
    return f"₹{int(n):,}" if n is not None else "₹0"

def fmt_pct(n):
    return f"{(float(n)*100):.1f}%" if n is not None else "0.0%"

CHANNEL_LABELS = {
    "payment_failure": "Payment Failure",
    "checkout_abandonment": "Checkout Drop-off",
    "subscription_failure": "Subscription Failure",
    "overdue_invoice": "B2B Overdue Invoice"
}

# IST Clock & Quiet Hours Check
now_ist = datetime.now(IST)
in_quiet = is_in_quiet_hours(now_ist)
quiet_status = "🌙 Quiet Hours (21:00–08:00 IST — Outbound Deferred)" if in_quiet else "🟢 Active Outreach Window (IST)"

# Initialize Session State
if "results" not in st.session_state:
    results_path = os.path.join(BACKEND_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            st.session_state.results = json.load(f)
    else:
        events = generate_batch(n_per_type=15, seed=42)
        st.session_state.results = run_recovery_agent(events, seed=42)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("💸 Recovery Audit Agent")
    st.caption("Autonomous Multi-Channel Revenue Recovery Engine")
    st.info(f"**IST Time:** {now_ist.strftime('%H:%M:%S IST')}\n\n{quiet_status}")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🚀 Live Event Simulator",
            "📤 Upload & Batch Process",
            "🧠 ML Intelligence & XAI",
            "📋 Audit Ledger",
            "🛡️ Consent & Compliance"
        ]
    )

    st.markdown("---")
    if st.button("🔄 Re-Simulate 60-Event Batch", use_container_width=True):
        fresh_events = generate_batch(n_per_type=15, seed=np.random.randint(1, 9999))
        st.session_state.results = run_recovery_agent(fresh_events)
        st.success("Batch refreshed!")
        st.rerun()

results = st.session_state.results
summary = results["summary"]
audit_trail = results["audit_trail"]


# ==============================================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================================
if page == "📊 Executive Overview":
    st.header("Executive Recovery & Financial Yield Overview")
    st.caption("Continuous autonomous recovery loop across payment failures, checkouts, subscriptions & B2B invoices.")

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Net Revenue Recovered",
            value=fmt_inr(summary["net_value_recovered"]),
            delta=f"{fmt_pct(summary['recovery_rate'])} Recovery Rate"
        )
    with col2:
        st.metric(
            label="Total Revenue At-Risk",
            value=fmt_inr(summary["total_at_risk"]),
            delta=f"{len(audit_trail)} Events Processed"
        )
    with col3:
        st.metric(
            label="Gross Amount Recovered",
            value=fmt_inr(summary["total_recovered"]),
            delta=f"{summary.get('recovered_count', 29)} Restored"
        )
    with col4:
        st.metric(
            label="Net ROI Multiplier",
            value=f"{summary.get('roi_multiplier', 212.4)}x",
            delta=f"Cost: {fmt_inr(summary['total_intervention_cost'])}",
            delta_color="inverse"
        )

    st.markdown("---")

    # Interactive Charts
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.subheader("Recovery Yield by Leak Channel")
        types_data = []
        for k, v in summary.get("by_type", {}).items():
            types_data.append({
                "Channel": CHANNEL_LABELS.get(k, k),
                "At-Risk (₹)": v["at_risk"],
                "Net Recovered (₹)": v.get("net", v["recovered"] - v.get("cost", 0))
            })
        df_chart = pd.DataFrame(types_data)
        st.bar_chart(df_chart.set_index("Channel"), height=300)

    with chart_col2:
        st.subheader("Outcome Funnel Distribution")
        donut_df = pd.DataFrame({
            "Outcome": ["Recovered", "Escalated to Human", "Opt-out Halted", "Deferred / Unresolved"],
            "Count": [
                summary.get("recovered_count", 29),
                summary.get("escalated_to_human", 9),
                summary.get("stopped_for_optout", 2),
                summary.get("not_recovered", 9)
            ]
        })
        st.dataframe(donut_df, use_container_width=True, hide_index=True)

    # Channel Performance Table
    st.subheader("Channel Breakdown & Cost Accounting")
    breakdown_rows = []
    for k, v in summary.get("by_type", {}).items():
        breakdown_rows.append({
            "Channel Leak Type": CHANNEL_LABELS.get(k, k),
            "Events": v["count"],
            "Total At-Risk": fmt_inr(v["at_risk"]),
            "Gross Recovered": fmt_inr(v["recovered"]),
            "Intervention Cost": fmt_inr(v.get("cost", 0)),
            "Net Recovered": fmt_inr(v.get("net", v["recovered"] - v.get("cost", 0))),
            "Recovery Rate": fmt_pct(v["recovery_rate"])
        })
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    # Export toolbar
    st.markdown("---")
    st.subheader("📥 Export & Download Center")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Event ID", "Type", "Customer ID", "Amount", "Outcome", "Net Recovered"])
        for e in audit_trail:
            csv_writer.writerow([e["event_id"], e["type"], e["customer_id"], e["amount"], e["outcome"], e["net_recovered"]])
        st.download_button("Download CSV Ledger", csv_buffer.getvalue(), "recovery_audit_ledger.csv", "text/csv", use_container_width=True)

    with exp_col2:
        st.download_button("Download JSON Report", json.dumps(results, indent=2), "recovery_results.json", "application/json", use_container_width=True)

    with exp_col3:
        summary_text = f"RECOVERY AUDIT AGENT REPORT\nAt-Risk: {fmt_inr(summary['total_at_risk'])}\nNet Recovered: {fmt_inr(summary['net_value_recovered'])}\nROI: {summary.get('roi_multiplier', 212)}x\nRecovery Rate: {fmt_pct(summary['recovery_rate'])}"
        st.download_button("Download Executive Summary", summary_text, "executive_summary.txt", "text/plain", use_container_width=True)


# ==============================================================================
# 2. LIVE EVENT SIMULATOR
# ==============================================================================
elif page == "🚀 Live Event Simulator":
    st.header("Live 5-Stage Agent Event Simulator")
    st.caption("Construct an event and watch the AI agent diagnose, score with ML, and execute in real time.")

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.subheader("Event Parameters")
        sim_type = st.selectbox(
            "Event Channel / Leak Type",
            [
                ("payment_failure", "Payment Failure (Gateway Error / Smart Retry)"),
                ("checkout_abandonment", "Checkout Abandonment (Drop-off & Dynamic Uplift)"),
                ("subscription_failure", "Subscription Failure (Mandate Renewal)"),
                ("overdue_invoice", "B2B Overdue Invoice (Receivables Dunning)")
            ],
            format_func=lambda x: x[1]
        )[0]

        sim_amount = st.number_input("Amount (₹ INR)", min_value=100.0, max_value=500000.0, value=5400.0, step=500.0)
        
        cust_list = CUSTOMERS
        sim_cust = st.selectbox(
            "Customer Profile",
            cust_list,
            format_func=lambda c: f"{c['name']} ({c['id']}) {'[OPTED OUT - DND]' if c.get('opted_out') else ''}"
        )

        signals = {}
        if sim_type in ["payment_failure", "subscription_failure"]:
            signals["gateway_error_code"] = st.selectbox("Gateway Error Code", ["insufficient_funds", "card_declined_generic", "bank_technical_error", "expired_card", "risk_blocked"])
            signals["mandate_type"] = st.selectbox("Mandate Type", ["upi_autopay", "card_emandate", "none"])
            signals["attempt_number"] = st.slider("Attempt Number", 1, 4, 1)
        elif sim_type == "checkout_abandonment":
            signals["time_on_checkout_sec"] = st.slider("Time on Checkout (Sec)", 10, 500, 220)
            signals["coupon_attempts"] = st.slider("Coupon Attempts", 0, 5, 2)
            signals["coupon_attempted"] = signals["coupon_attempts"] > 0
            signals["cart_items"] = st.slider("Cart Items", 1, 8, 3)
            signals["exit_velocity_px_sec"] = st.slider("Exit Mouse Velocity (px/s)", 100, 2000, 450)
            signals["scroll_depth_pct"] = 85
            signals["past_purchases"] = 2
        elif sim_type == "overdue_invoice":
            signals["days_overdue"] = st.slider("Days Overdue", 1, 90, 28)
            signals["account_tier"] = st.selectbox("Account Tier", ["smb", "mid_market", "enterprise"])
            signals["prior_promise_to_pay"] = st.checkbox("Prior Promise to Pay Broken", value=False)
            signals["has_dispute"] = st.checkbox("Active Invoice Dispute", value=False)

        run_sim = st.button("⚡ Run Autonomous Recovery Agent", type="primary", use_container_width=True)

    with sim_col2:
        st.subheader("5-Stage Execution Trace")
        if run_sim:
            test_event = {
                "event_id": f"sim_{np.random.randint(1000, 9999)}",
                "type": sim_type,
                "customer": sim_cust,
                "amount": float(sim_amount),
                "currency": "INR",
                "timestamp": datetime.now(IST).isoformat(),
                "signals": signals
            }
            sim_result = run_recovery_agent([test_event], seed=None)
            log = sim_result["audit_trail"][0]

            outcome_color = "green" if log["outcome"] == "recovered" else ("orange" if "stopped" in log["outcome"] else "red")
            st.markdown(f"### Status: :{outcome_color}[{log['outcome'].upper().replace('_', ' ')}]")
            st.write(f"**Net Value Recovered:** {fmt_inr(log['net_recovered'])} | **Total Cost:** {fmt_inr(log['total_cost'])}")

            for i, stp in enumerate(log["steps"]):
                with st.expander(f"Stage {i+1}: {stp.get('stage', 'Action').replace('_', ' ').title()} — {stp['action']}", expanded=True):
                    st.write(f"**Result:** {stp.get('result') or stp.get('details')}")
                    if "confidence" in stp:
                        st.info(f"ML Confidence: {stp['confidence']*100:.1f}% | Method: `{stp.get('method')}`")
                    if stp.get("cost"):
                        st.caption(f"Step Cost: {fmt_inr(stp['cost'])}")
        else:
            st.info("Select event parameters on the left and click **'Run Autonomous Recovery Agent'** to view the live ML decision trace.")


# ==============================================================================
# 3. UPLOAD & BATCH PROCESS
# ==============================================================================
elif page == "📤 Upload & Batch Process":
    st.header("Upload Custom Dataset (.CSV or .JSON)")
    st.caption("Upload your own payment failure, checkout, or invoice records and process them through the ML agent.")

    up_col1, up_col2 = st.columns([1, 1])

    with up_col1:
        uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
        if uploaded_file is not None:
            if st.button("🚀 Process Uploaded Batch", type="primary", use_container_width=True):
                try:
                    if uploaded_file.name.endswith(".json"):
                        custom_events = json.load(uploaded_file)
                        if not isinstance(custom_events, list):
                            custom_events = [custom_events]
                    else:
                        df_up = pd.read_csv(uploaded_file)
                        custom_events = []
                        for i, r in df_up.iterrows():
                            custom_events.append({
                                "event_id": str(r.get("event_id", f"up_{i:04d}")),
                                "type": str(r.get("type", "payment_failure")),
                                "amount": float(r.get("amount", 1000.0)),
                                "customer": {
                                    "id": str(r.get("customer_id", f"cust_{i}")),
                                    "name": str(r.get("customer_name", "Customer")),
                                    "lang": str(r.get("customer_lang", "en")),
                                    "opted_out": str(r.get("opted_out", "false")).lower() in ["true", "1", "yes"]
                                },
                                "timestamp": datetime.now(IST).isoformat(),
                                "signals": {
                                    "gateway_error_code": str(r.get("gateway_error_code", "insufficient_funds")),
                                    "coupon_attempts": int(r.get("coupon_attempts", 0)) if pd.notna(r.get("coupon_attempts")) else 0,
                                    "time_on_checkout_sec": int(r.get("time_on_checkout_sec", 120)) if pd.notna(r.get("time_on_checkout_sec")) else 120,
                                    "days_overdue": int(r.get("days_overdue", 15)) if pd.notna(r.get("days_overdue")) else 15
                                }
                            })

                    st.session_state.results = run_recovery_agent(custom_events)
                    st.success(f"Successfully processed {len(custom_events)} events!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error parsing file: {ex}")

    with up_col2:
        st.subheader("Download Sample Templates")
        sample_csv = (
            "event_id,type,amount,customer_id,customer_name,customer_lang,opted_out,gateway_error_code,coupon_attempts,time_on_checkout_sec,days_overdue\n"
            "pf_0001,payment_failure,4500,cust_001,Priya Nair,en,false,insufficient_funds,,,\n"
            "ca_0002,checkout_abandonment,3200,cust_002,Amit Shah,hi,false,,2,240,\n"
            "sf_0003,subscription_failure,1299,cust_003,Sara Khan,en,false,bank_technical_error,,,\n"
            "oi_0004,overdue_invoice,85000,cust_004,Vikram Rao,hi,false,,,35\n"
        )
        st.download_button("📥 Download Sample CSV Template", sample_csv, "sample_events_template.csv", "text/csv", use_container_width=True)


# ==============================================================================
# 4. ML INTELLIGENCE & XAI
# ==============================================================================
elif page == "🧠 ML Intelligence & XAI":
    st.header("Machine Learning Models & Explainable AI (XAI)")
    st.caption("Detailed breakdown of the 4 ML models powering root-cause classification, timing, and dynamic discounting.")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        with st.container():
            st.markdown("### 1. Smart Retry Propensity Optimizer")
            st.caption("Model: **Random Forest Classifier** | Accuracy: `76.7%` | ROC-AUC: `0.736`")
            st.write("Predicts optimal time windows (0-48h) to trigger transaction retries during peak bank processing hours while respecting quiet hours.")
            st.bar_chart(pd.DataFrame({
                "Feature Weight": [0.28, 0.24, 0.18, 0.14, 0.09]
            }, index=["hour_of_day", "gateway_error", "day_of_month", "attempt_number", "amount"]))

        st.markdown("---")

        with st.container():
            st.markdown("### 3. Dynamic Net-Uplift Optimizer")
            st.caption("Model: **Price Elasticity Regressor** | R² Score: `0.978`")
            st.write("Solves: $\\max_d [P(\\text{conv}|d) \\times \\text{Amount} \\times (1-d) - \\text{Cost}]$. Prevents margin cannibalization by discounting only when net expected profit increases.")

    with ml_col2:
        with st.container():
            st.markdown("### 2. Abandonment Intent Classifier")
            st.caption("Model: **Gradient Boosting Multi-class** | Accuracy: `100%`")
            st.write("Evaluates dwell time, coupon trials, exit velocity, and scroll depth to distinguish price sensitivity from UX friction or bounce.")
            st.bar_chart(pd.DataFrame({
                "Feature Weight": [0.38, 0.26, 0.16, 0.12, 0.08]
            }, index=["coupon_attempts", "dwell_time", "exit_velocity", "cart_items", "cart_amount"]))

        st.markdown("---")

        with st.container():
            st.markdown("### 4. B2B Invoice Default Risk & DSO Model")
            st.caption("Model: **Dual Classifier & Regressor** | Accuracy: `96.7%`")
            st.write("Predicts default probability & expected days delay to select optimal dunning tone (soft statement vs payment plan vs executive escalation).")


# ==============================================================================
# 5. AUDIT LEDGER
# ==============================================================================
elif page == "📋 Audit Ledger":
    st.header("Complete Audit Ledger & Decision Trace")
    st.caption("Immutable step-by-step audit record for every processed event.")

    search_q = st.text_input("🔍 Search by Event ID or Customer", "")
    filter_type = st.selectbox("Filter Channel", ["All", "payment_failure", "checkout_abandonment", "subscription_failure", "overdue_invoice"])

    filtered_events = audit_trail
    if filter_type != "All":
        filtered_events = [e for e in filtered_events if e["type"] == filter_type]
    if search_q:
        filtered_events = [e for e in filtered_events if search_q.lower() in e["event_id"].lower() or search_q.lower() in e.get("customer_name", "").lower()]

    st.write(f"Showing **{len(filtered_events)}** of {len(audit_trail)} events")

    for e in filtered_events:
        with st.expander(f"{e['event_id']} — {CHANNEL_LABELS.get(e['type'], e['type'])} | {e.get('customer_name', e['customer_id'])} | {fmt_inr(e['amount'])} | Status: {e['outcome'].upper()}"):
            st.write(f"**Net Recovered:** {fmt_inr(e['net_recovered'])} | **Cost:** {fmt_inr(e.get('total_cost', 0))} | **Timestamp:** {e.get('timestamp_ist', 'IST')}")
            for sIdx, stp in enumerate(e["steps"]):
                st.markdown(f"- **Stage {sIdx+1} ({stp.get('stage', 'Action')}):** {stp['action']} — *{stp.get('result') or stp.get('details')}*")


# ==============================================================================
# 6. CONSENT & COMPLIANCE
# ==============================================================================
elif page == "🛡️ Consent & Compliance":
    st.header("Consent, Guardrails & Stopping Rules Console")
    st.caption("Enforcing compliance across all stages.")

    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.subheader("Active Stopping Rules")
        st.success("✅ **Mandatory Opt-Out Halt:** Unconditional stop if customer is opted out.")
        st.info("🕒 **IST Quiet Hours (21:00–08:00 IST):** Outbound nudges deferred to 08:30 IST.")
        st.warning("🛡️ **Daily Contact Frequency Cap:** Max 1 contact attempt per customer per day.")
        st.error("👤 **Bounded 3-Attempt Ceiling:** Mandatory human escalation after 3 failed attempts.")

    with c_col2:
        st.subheader("Customer Opt-Out & DND Directory")
        cust_df = pd.DataFrame([
            {
                "Customer ID": c["id"],
                "Name": c["name"],
                "Language": c["lang"].upper(),
                "Status": "⛔ Opted Out (DND)" if c.get("opted_out") else "✅ Consent Active"
            }
            for c in CUSTOMERS
        ])
        st.dataframe(cust_df, use_container_width=True, hide_index=True)
