"""
server.py — Enterprise Revenue Recovery API & Live Simulator Backend
Exposes REST endpoints for real-time event ingestion, batch simulations,
file upload (JSON/CSV), results export (CSV/JSON), ML diagnostics, and dashboard.
"""

import os
import sys
import json
import csv
import io
from flask import Flask, request, jsonify, send_from_directory, Response

# Ensure backend and backend/ml in sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BACKEND_DIR, "ml")
DASHBOARD_DIR = os.path.join(BACKEND_DIR, "..", "dashboard")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from agent import run_recovery_agent, diagnose_event, build_intervention_ladder
from generate_data import generate_batch, CUSTOMERS, IST
from ml import load_all_models

app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="")


# Enable CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# Cached latest batch results
LATEST_RESULTS = None


def get_or_create_results():
    global LATEST_RESULTS
    if LATEST_RESULTS is None:
        results_path = os.path.join(BACKEND_DIR, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                LATEST_RESULTS = json.load(f)
        else:
            events = generate_batch(n_per_type=15, seed=42)
            LATEST_RESULTS = run_recovery_agent(events, seed=42)
    return LATEST_RESULTS


def parse_csv_events(csv_text):
    """Parses arbitrary CSV into normalized recovery agent events."""
    reader = csv.DictReader(io.StringIO(csv_text))
    events = []
    for i, row in enumerate(reader):
        etype = row.get("type") or row.get("event_type") or "payment_failure"
        amount = float(row.get("amount") or 1000.0)
        cid = row.get("customer_id") or f"cust_{i:03d}"
        cname = row.get("customer_name") or row.get("name") or "Customer"
        lang = row.get("customer_lang") or row.get("lang") or "en"
        opted_out = str(row.get("opted_out", "false")).lower() in ["true", "1", "yes"]

        # Parse signals
        signals = {}
        for k, v in row.items():
            if k in ["event_id", "type", "event_type", "amount", "customer_id", "customer_name", "name", "lang", "customer_lang", "opted_out"]:
                continue
            # Try type conversion
            try:
                if v.isdigit():
                    signals[k] = int(v)
                elif v.replace(".", "", 1).isdigit():
                    signals[k] = float(v)
                elif v.lower() in ["true", "false"]:
                    signals[k] = v.lower() == "true"
                else:
                    signals[k] = v
            except Exception:
                signals[k] = v

        if "coupon_attempted" not in signals and "coupon_attempts" in signals:
            signals["coupon_attempted"] = signals["coupon_attempts"] > 0

        events.append({
            "event_id": row.get("event_id") or f"evt_{i+1:04d}",
            "type": etype,
            "customer": {
                "id": cid,
                "name": cname,
                "lang": lang,
                "opted_out": opted_out,
            },
            "amount": amount,
            "currency": row.get("currency", "INR"),
            "timestamp": row.get("timestamp"),
            "signals": signals,
        })
    return events


# -------------------------------------------------------------
# STATIC DASHBOARD ROUTING
# -------------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(DASHBOARD_DIR, filename)


# -------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def get_status():
    models = load_all_models()
    return jsonify({
        "status": "online",
        "agent_name": "Recovery Audit Agent (Enterprise ML Edition)",
        "models_active": list(models.keys()),
        "timezone": "Asia/Kolkata (IST, UTC+5:30)",
        "stopping_rules": [
            "Mandatory Opt-Out Halt",
            "Quiet Hours Deferral (21:00–08:00 IST)",
            "Same-day Customer Contact Frequency Capping",
            "Max 3 Automated Attempts before Human Escalation",
        ]
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    res = get_or_create_results()
    return jsonify(res["summary"])


@app.route("/api/audit", methods=["GET"])
def get_audit():
    res = get_or_create_results()
    etype = request.args.get("type", "all")
    outcome = request.args.get("outcome", "all")
    search = request.args.get("q", "").lower()

    filtered = res["audit_trail"]
    if etype != "all":
        filtered = [e for e in filtered if e["type"] == etype]
    if outcome != "all":
        filtered = [e for e in filtered if e["outcome"] == outcome]
    if search:
        filtered = [
            e for e in filtered
            if search in e["event_id"].lower()
            or search in e["customer_id"].lower()
            or search in e.get("customer_name", "").lower()
        ]

    return jsonify({
        "total": len(res["audit_trail"]),
        "filtered_count": len(filtered),
        "audit_trail": filtered,
    })


@app.route("/api/batch/run", methods=["POST"])
def run_batch():
    global LATEST_RESULTS
    data = request.get_json() or {}
    n_per_type = int(data.get("n_per_type", 15))
    seed = int(data.get("seed", 42))

    events = generate_batch(n_per_type=n_per_type, seed=seed)
    LATEST_RESULTS = run_recovery_agent(events, seed=seed)

    # Save to disk
    results_path = os.path.join(BACKEND_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(LATEST_RESULTS, f, indent=2)

    dashboard_data_path = os.path.join(DASHBOARD_DIR, "data.json")
    if os.path.exists(DASHBOARD_DIR):
        with open(dashboard_data_path, "w") as f:
            json.dump(LATEST_RESULTS, f, indent=2)

    return jsonify(LATEST_RESULTS["summary"])


@app.route("/api/batch/upload", methods=["POST"])
def upload_batch():
    """
    Accepts custom event dataset via file upload (.json or .csv) or JSON body.
    Processes all events through the ML Agent pipeline and returns fresh results.
    """
    global LATEST_RESULTS
    events = []

    if "file" in request.files:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename.lower()
        content = uploaded_file.read().decode("utf-8")

        if filename.endswith(".json"):
            events = json.loads(content)
            if not isinstance(events, list):
                events = [events]
        elif filename.endswith(".csv"):
            events = parse_csv_events(content)
        else:
            return jsonify({"error": "Unsupported file format. Please upload .json or .csv"}), 400
    elif request.is_json:
        data = request.get_json()
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]
        else:
            events = [data]
    else:
        return jsonify({"error": "No file or JSON payload received"}), 400

    if not events:
        return jsonify({"error": "No valid events found in upload"}), 400

    # Run agent loop
    LATEST_RESULTS = run_recovery_agent(events, seed=None)

    # Persist
    results_path = os.path.join(BACKEND_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(LATEST_RESULTS, f, indent=2)

    dashboard_data_path = os.path.join(DASHBOARD_DIR, "data.json")
    if os.path.exists(DASHBOARD_DIR):
        with open(dashboard_data_path, "w") as f:
            json.dump(LATEST_RESULTS, f, indent=2)

    return jsonify({
        "status": "success",
        "events_processed": len(events),
        "summary": LATEST_RESULTS["summary"],
        "audit_trail": LATEST_RESULTS["audit_trail"]
    })


@app.route("/api/event/simulate", methods=["POST"])
def simulate_single_event():
    """
    Simulates a live event through the complete 5-stage agent pipeline in real time.
    """
    event = request.get_json()
    if not event:
        return jsonify({"error": "Event payload required"}), 400

    single_result = run_recovery_agent([event], seed=None)
    log = single_result["audit_trail"][0]
    return jsonify(log)


@app.route("/api/export/csv", methods=["GET"])
def export_audit_csv():
    """Exports full audit ledger as a downloadable CSV spreadsheet."""
    res = get_or_create_results()
    output = io.StringIO()
    writer = csv.writer(output)

    # CSV Header
    writer.writerow([
        "Event ID", "Type", "Customer ID", "Customer Name", "Amount (INR)",
        "Diagnosed Cause", "Confidence", "Method", "Outcome", "Intervention Cost (INR)",
        "Net Recovered (INR)", "Timestamp (IST)", "Execution Steps Summary"
    ])

    for e in res["audit_trail"]:
        diag = next((st for st in e["steps"] if st.get("stage") == "diagnosis"), {})
        steps_summary = " -> ".join([
            f"{st.get('action')}({st.get('result', '').split('—')[0].strip()})"
            for st in e["steps"]
        ])
        writer.writerow([
            e.get("event_id"),
            e.get("type"),
            e.get("customer_id"),
            e.get("customer_name"),
            e.get("amount"),
            diag.get("cause", ""),
            diag.get("confidence", ""),
            diag.get("method", ""),
            e.get("outcome"),
            e.get("total_cost", 0),
            e.get("net_recovered", 0),
            e.get("timestamp_ist", ""),
            steps_summary
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=revenue_recovery_audit_ledger.csv"}
    )


@app.route("/api/template/csv", methods=["GET"])
def download_sample_csv():
    """Provides a sample CSV template for user upload."""
    sample_csv = (
        "event_id,type,amount,customer_id,customer_name,customer_lang,opted_out,gateway_error_code,coupon_attempts,time_on_checkout_sec,days_overdue\n"
        "pf_0001,payment_failure,4500,cust_001,Priya Nair,en,false,insufficient_funds,,,\n"
        "ca_0002,checkout_abandonment,3200,cust_002,Amit Shah,hi,false,,2,240,\n"
        "sf_0003,subscription_failure,1299,cust_003,Sara Khan,en,false,bank_technical_error,,,\n"
        "oi_0004,overdue_invoice,85000,cust_004,Vikram Rao,hi,false,,,35\n"
        "pf_0005,payment_failure,7800,cust_007,Fatima Sheikh,hi,true,card_declined_generic,,,\n"
    )
    return Response(
        sample_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sample_events_template.csv"}
    )


@app.route("/api/ml/insights", methods=["GET"])
def get_ml_insights():
    insights_path = os.path.join(ML_DIR, "models", "ml_insights.json")
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Insights not yet generated"}), 404


@app.route("/api/customers", methods=["GET"])
def get_customers():
    return jsonify(CUSTOMERS)


@app.route("/api/customer/consent", methods=["POST"])
def toggle_consent():
    data = request.get_json() or {}
    cid = data.get("customer_id")
    opted_out = data.get("opted_out", True)

    for c in CUSTOMERS:
        if c["id"] == cid:
            c["opted_out"] = opted_out
            return jsonify({"status": "updated", "customer": c})

    return jsonify({"error": "Customer not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Revenue Recovery Agent Server on http://localhost:{port} ...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
