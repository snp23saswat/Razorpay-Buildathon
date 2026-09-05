# Blueprint: Autonomous Revenue Recovery Platform (Enterprise ML Edition)

This document is the technical architecture and decision specification behind the build. Use this as your defense document for technical panels and review rounds.

---

## 1. Executive Summary & Core Hypothesis

Revenue leaks through four different operational channels — failed payments, abandoned checkouts, failed subscriptions, and overdue invoices. On the surface, they appear to require separate point solutions. In reality, they share the exact same underlying mechanics: **identifying at-risk expected cash flow, diagnosing root causes, determining the optimal intervention bounded by compliance guardrails, and measuring Net Value Recovered (Gross Recovery minus Intervention Costs).**

This platform establishes **one unified autonomous loop across all four channels**, deploying Machine Learning precisely where statistical inference and price elasticity deliver measurable ROI over naive heuristics.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    A[Event Sources] -->|payment.failed| B[Event Normalizer]
    A -->|checkout.abandoned| B
    A -->|subscription.failed| B
    A -->|invoice.overdue| B

    B --> C{Stage 1: Consent Gate}
    C -->|Opted Out / DND| STOP1[Halt — Logged, Zero Contact]
    C -->|Authorized| C2{Stage 2: Frequency Cap Gate}
    C2 -->|Already Contacted Today| DEFER1[Defer to Tomorrow's Window]
    C2 -->|OK| D[Stage 3: Hybrid Diagnosis Engine]

    D -->|Gateway Error Signal| D1[Deterministic Mapping + ML Propensity]
    D -->|Checkout Session Telemetry| D2[ML Behavioral Intent Classifier]
    D -->|B2B Receivables Aging| D3[ML Invoice Default Risk Regressor]

    D1 --> E[Stage 4: ML-Guided Policy Engine]
    D2 --> E1[ML Dynamic Net-Uplift Optimizer]
    E1 --> E
    D3 --> E

    E --> F{Stage 5: Bounded Intervention Ladder}
    F --> G[IST Quiet-Hours Check (21:00–08:00 IST)]
    G -->|Inside Quiet Window| DEFER2[Reschedule for 08:30 IST]
    G -->|Active Window| H[Autonomous Execution & Smart Retry]

    H -->|Recovered| I[Mark Recovered & Log Net ROI]
    H -->|Failed + Attempts < 3| F
    H -->|Attempts Exhausted| J[Mandatory Human Escalation]

    I --> K[Explainable Audit Ledger]
    J --> K
    STOP1 --> K
    DEFER1 --> K
    DEFER2 --> K
```

---

## 3. Machine Learning Decision Subsystems

### 3.1 Model 1: Smart Retry Propensity & Timing Optimizer (`ml_smart_retry.py`)
- **Problem**: Fixed cooldowns and uniform retries ignore bank clearing cycles and user pay-cycles, causing payment fatigue and gateway decline fees.
- **Model**: Random Forest Classifier trained on multidimensional transaction features (`amount`, `gateway_error_code`, `mandate_type`, `attempt_number`, `hour_of_day`, `day_of_month`).
- **Policy Decision**: Evaluates candidate retry offsets (+2h, +4h, +8h, +12h, +24h, +48h) and targets peak banking clearance hours (9-11 AM, 5-8 PM IST) while steering clear of IST quiet hours.

### 3.2 Model 2: Abandonment Intent & Behavioral Classifier (`ml_intent_classifier.py`)
- **Problem**: Naive keyword/rule heuristics cannot distinguish between genuine price sensitivity, hesitation over shipping/trust, UI glitches, or casual bounces.
- **Model**: Calibrated Gradient Boosting Multi-class Classifier.
- **Features**: `dwell_time_sec`, `coupon_attempt_count`, `cart_item_count`, `cart_amount`, `scroll_depth_pct`, `exit_velocity_px_sec`, `past_purchase_frequency`.
- **Outputs**: Probabilistic intent vector across `price_sensitivity`, `friction_hesitation`, `technical_issue`, and `distraction_or_bounce` with Explainable AI (XAI) feature attributions.

### 3.3 Model 3: Dynamic Discount & Net-Uplift Optimizer (`ml_uplift_optimizer.py`)
- **Problem**: Blanket discounting recovers revenue at the expense of severe margin destruction (giving discounts to users who would have converted anyway).
- **Model**: Price Elasticity Regressor predicting conversion probability given discount tier $d \in \{0\%, 5\%, 10\%, 15\%\}$.
- **Objective Function**:
  $$\max_{d} \mathbb{E}[\text{Net Revenue}] = \left[ P(\text{conversion} \mid d, X) \times \text{Amount} \times (1 - d) \right] - \text{Execution Cost}$$
- **Result**: Only grants incentives when the incremental conversion uplift outweighs the discount cost.

### 3.4 Model 4: B2B Invoice Default Risk & DSO Regressor (`ml_invoice_risk.py`)
- **Problem**: Static overdue age rules treat strategic enterprise partners the same as high-risk defaulters.
- **Model**: Dual Random Forest Classifier and Gradient Boosting Regressor evaluating invoice aging, dispute history, and client relationship length to predict default probability and expected days-to-pay.
- **Policy Decision**: Selects customized dunning tone (courtesy account statement vs structured payment plan vs executive escalation).

---

## 4. Compliance Guardrails & Safety Limits

1. **Unconditional Consent First**: If a customer has opted out, all execution ceases immediately.
2. **IST Quiet Hours**: 21:00 to 08:00 IST is protected. Outbound messages are deferred to the 08:30 IST window.
3. **Daily Contact Frequency Limit**: Maximum 1 outbound contact per customer per calendar day.
4. **Hard 3-Attempt Ceiling**: The agent is strictly bounded. After 3 automated attempts, tickets are transferred to human specialists.
5. **Cost-Aware Optimization**: Every attempt incurs an explicit simulated cost (gateway fees, WhatsApp API, discount %, agent hourly cost). Net Value Recovered is the primary success metric.

---

## 5. Auditability & Explainable AI (XAI)

Every event produces an immutable audit record containing:
- Exact diagnosis method (Deterministic Rule, Gradient Boosting, Random Forest)
- Model confidence scores and top contributing behavioral signals
- Timestamp in IST
- Itemized intervention cost
- Net financial yield
