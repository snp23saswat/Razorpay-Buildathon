Markdown# Revenue Recovery Agent (Enterprise ML Edition)

> **Autonomous Multi-Channel Revenue Recovery Engine powered by Machine Learning, Dynamic Uplift Optimization, and Explainable AI.**

---

## 1. Problem & Architecture

Revenue leaks through four distinct doors — **payment failures**, **abandoned checkouts**, **failed subscription renewals**, and **B2B overdue receivables**. While the symptom looks different in each channel, the underlying objective is identical: **recovering at-risk money while strictly optimizing Net Value and adhering to compliance guardrails**.

```mermaid
flowchart TD
    A[Event Ingestion Stream] --> B{Stage 1: Consent Gate}
    B -->|Opted Out / DND| STOP1[Halt — Logged, Zero Contact]
    B -->|Authorized| C{Stage 2: Deduplication Gate}
    C -->|Frequency Limit Reached| DEFER1[Defer to Next Day]
    C -->|OK| D[Stage 3: Hybrid Diagnosis Engine]

    D -->|Gateway Codes| D1["Deterministic Mapping + ML Propensity"]
    D -->|Checkout Session| D2["ML Intent Classifier (Gradient Boosting)"]
    D -->|B2B Receivables| D3["ML Invoice Default Risk Regressor"]

    D1 --> E[Stage 4: ML-Guided Policy Engine]
    D2 --> E1[ML Dynamic Uplift Optimizer]
    E1 --> E
    D3 --> E

    E --> F{Stage 5: Bounded Intervention Ladder}
    F --> G["IST Quiet-Hours Gate (21:00-08:00 IST)"]
    G -->|Inside Quiet Window| DEFER2[Reschedule for 08:30 IST]
    G -->|Outreach Window| H[Autonomous Execution & Smart Retry]

    H -->|Recovered| I[Mark Recovered & Calculate Net Yield]
    H -->|Fail + Attempts < 3| F
    H -->|Attempts Exhausted| J[Mandatory Human Escalation]

    I --> K[Explainable Audit Ledger]
    J --> K
    STOP1 --> K
    DEFER1 --> K
    DEFER2 --> K
2. The 4 Machine Learning SubsystemsModelTechniqueTarget DecisionROI Impact1. Smart Retry OptimizerRandom Forest Classifier (ml_smart_retry.py)Predicts recovery propensity across time windows (0-48h), targeting peak bank throughput (9-11 AM, 5-8 PM)Maximizes payment success probability while avoiding unnecessary retry fees.2. Abandonment Intent ClassifierGradient Boosting Multi-class (ml_intent_classifier.py)Evaluates session telemetry (dwell_time, coupon_attempts, scroll_depth, exit_velocity) to classify intent (price_sensitivity, friction_hesitation, technical_issue, bounce)Eliminates guesswork and triggers targeted interventions.3. Net-Uplift Incentive OptimizerRandom Forest Price-Elasticity Regressor (ml_uplift_optimizer.py)Evaluates candidate discounts (0%, 5%, 10%, 15%) to solve: $\max_d [P(\text{conv} \mid d) \cdot \text{Amount} \cdot (1-d) - \text{Cost}]$Prevents margin cannibalization by offering discounts only when incremental net yield is positive.4. B2B Receivables Risk ModelDual Classifier & Regressor (ml_invoice_risk.py)Evaluates DSO, account tier, and dispute signals to predict default risk score & expected payment delayCustomizes dunning tone: soft reminder vs structured payment plan vs executive escalation.3. Strict Compliance & Stopping RulesUnconditional Consent Check: Opted-out customers are halted before any diagnosis or contact.IST Quiet Hours (21:00–08:00 IST): Outbound nudges during nighttime are queued and rescheduled for the morning active window (08:30 IST).Same-Day Customer Frequency Capping: Customers are contacted at most once per calendar day to avoid notification fatigue.Bounded 3-Attempt Ceiling: The agent never loops indefinitely; mandatory human escalation after 3 automated attempts.Net Value Scoring: Measures recovery net of all intervention costs (discounts, gateway fees, agent time).4. Benchmark Performance (60-Event Batch)Total Revenue At-Risk: ₹29.08 LakhGross Recovered: ₹19.66 LakhTotal Intervention Cost: ₹8,331Net Value Recovered: ₹19.58 LakhNet ROI Multiplier: 235xOverall Recovery Rate: 67.6%Compliance Halts: 2 Opt-Out stops, 27 Quiet-Hour deferrals, 20 Frequency deduplications, 9 Human escalations.5. Quick Start & Execution1. Train ML Models & Run Core Agent:Bash# Train ML models and export serializations
python backend/ml/train_models.py

# Run end-to-end recovery agent
python backend/agent.py
2. Launch Interactive Web Server & Dashboard:Bashpython backend/server.py
Open your browser at http://localhost:5000 to access:Executive Overview: Real-time KPI cards & Chart.js channel breakdowns.Live Event Simulator: Interactive testbed to simulate single events and view live 5-stage ML execution traces.ML Intelligence & XAI Inspector: Feature importances, model metrics, and explainability bars.Audit Ledger: Searchable, expandable trace view with exact timestamps and financial breakdowns.Consent & Compliance Console: Live IST clock, quiet-hours monitor, and 1-click opt-out toggle.6. Project Structurerevenue-recovery-agent/
├── backend/
│   ├── agent.py                 # 5-stage core loop + policy engine
│   ├── generate_data.py         # High-dimensional synthetic event stream generator
│   ├── server.py                # Flask REST API + live simulator backend
│   ├── results.json             # Execution results & audit trail
│   └── ml/
│       ├── __init__.py          # Model loader & registry
│       ├── ml_smart_retry.py    # Model 1: Smart Retry Propensity Optimizer
│       ├── ml_intent_classifier.py # Model 2: Abandonment Intent Classifier
│       ├── ml_uplift_optimizer.py  # Model 3: Dynamic Discount & Net Uplift Optimizer
│       ├── ml_invoice_risk.py   # Model 4: B2B Invoice Risk & DSO Regressor
│       ├── train_models.py      # Training & serialization pipeline
│       └── models/              # Serialized .joblib artifacts & ml_insights.json
├── dashboard/
│   ├── index.html               # Multi-tab responsive web application
│   ├── style.css                # Glassmorphic dark fintech design system
│   ├── app.js                   # Client controller (REST API + data.json fallback)
│   └── data.json                # Pre-rendered snapshot for offline viewing
├── BLUEPRINT.md                 # Technical specification and panel defense document
└── README.md                    # Project overview & documentation
