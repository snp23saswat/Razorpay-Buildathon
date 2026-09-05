"""
generate_data.py — Enterprise Revenue Recovery Event Generator
Generates realistic high-dimensional synthetic event streams across all 4 leak channels:
  1. payment_failure        (Gateway degradation -> smart retry & recovery optimization)
  2. checkout_abandonment   (Drop-off telemetry -> ML intent classification & dynamic uplift)
  3. subscription_failure   (Recurring charge failure -> mandate recovery & dunning)
  4. overdue_invoice        (B2B receivables -> risk scoring & personalized dunning)

Each event contains structured identifiers, timestamps (IST-aware), customer profiles,
and behavioral telemetry for both rule-based and machine-learning decision engines.
"""

import json
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

# IST Timezone (UTC + 5:30)
IST = timezone(timedelta(hours=5, minutes=30))

CUSTOMERS = [
    {"id": f"cust_{i:03d}", "name": n, "lang": lang, "opted_out": False, "tenure_months": tenure, "tier": tier}
    for i, (n, lang, tenure, tier) in enumerate([
        ("Rahul Verma", "hi", 18, "standard"),
        ("Priya Nair", "en", 34, "vip"),
        ("Amit Shah", "hi", 6, "standard"),
        ("Sara Khan", "en", 24, "vip"),
        ("Vikram Rao", "hi", 4, "standard"),
        ("Neha Gupta", "en", 42, "vip"),
        ("Karthik Iyer", "en", 12, "standard"),
        ("Fatima Sheikh", "hi", 8, "standard"),  # Explicit opt-out for compliance validation
        ("Arjun Mehta", "en", 30, "vip"),
        ("Divya Menon", "hi", 15, "standard"),
        ("Rohan Das", "en", 3, "standard"),
        ("Anjali Singh", "hi", 28, "vip"),
        ("Siddharth Joshi", "en", 20, "vip"),
        ("Pooja Sharma", "hi", 9, "standard"),
        ("Manoj Kulkarni", "en", 14, "standard"),
    ])
]

# Customer #7 has explicitly opted out — stopping rule must unconditionally halt contact
CUSTOMERS[7]["opted_out"] = True

PAYMENT_ERROR_CODES = [
    ("insufficient_funds", 0.35),
    ("card_declined_generic", 0.25),
    ("bank_technical_error", 0.15),
    ("expired_card", 0.15),
    ("risk_blocked", 0.10),
]

CARD_TYPES = ["visa_credit", "mastercard_debit", "rupay_debit", "amex_credit"]
MANDATE_TYPES = ["upi_autopay", "card_emandate", "netbanking", "none"]
DEVICES = ["mobile", "desktop", "tablet"]


def pick_weighted(pairs):
    r = random.random()
    cum = 0
    for val, w in pairs:
        cum += w
        if r <= cum:
            return val
    return pairs[-1][0]


def gen_payment_failure(i, base_time=None):
    if base_time is None:
        base_time = datetime.now(IST)
    cust = random.choice(CUSTOMERS)
    ts = base_time - timedelta(hours=random.randint(0, 48), minutes=random.randint(0, 59))
    error_code = pick_weighted(PAYMENT_ERROR_CODES)
    attempt_num = random.randint(1, 3)

    return {
        "event_id": f"pf_{i:04d}",
        "type": "payment_failure",
        "customer": cust,
        "amount": round(random.uniform(299, 18500), 2),
        "currency": "INR",
        "timestamp": ts.isoformat(),
        "signals": {
            "gateway_error_code": error_code,
            "attempt_number": attempt_num,
            "mandate_type": random.choice(MANDATE_TYPES),
            "card_type": random.choice(CARD_TYPES),
            "customer_tenure_months": cust.get("tenure_months", 12),
            "hour_of_day": ts.hour,
            "day_of_month": ts.day,
            "bank_gateway_latency_ms": random.randint(120, 1800),
        },
    }


def gen_checkout_abandonment(i, base_time=None):
    if base_time is None:
        base_time = datetime.now(IST)
    cust = random.choice(CUSTOMERS)
    ts = base_time - timedelta(hours=random.randint(0, 36), minutes=random.randint(0, 59))

    # Latent behavioral archetype
    intent_archetype = random.choices(
        ["price_sensitivity", "friction_hesitation", "technical_issue", "distraction_or_bounce"],
        weights=[0.35, 0.30, 0.15, 0.20]
    )[0]

    cart_amount = round(random.uniform(499, 12500), 2)
    past_purchases = random.randint(0, 10)

    if intent_archetype == "price_sensitivity":
        coupon_attempts = random.randint(1, 4)
        time_on_sec = random.randint(90, 420)
        cart_items = random.randint(2, 6)
        scroll_depth = random.randint(65, 100)
        exit_velocity = random.randint(250, 750)
        payment_selected = random.choice(["upi", "card", "none"])
    elif intent_archetype == "friction_hesitation":
        coupon_attempts = 0
        time_on_sec = random.randint(150, 600)
        cart_items = random.randint(1, 5)
        scroll_depth = random.randint(80, 100)
        exit_velocity = random.randint(150, 450)
        payment_selected = random.choice(["netbanking", "card", "none"])
    elif intent_archetype == "technical_issue":
        coupon_attempts = random.choice([0, 1])
        time_on_sec = random.randint(45, 180)
        cart_items = random.randint(1, 3)
        scroll_depth = random.randint(40, 75)
        exit_velocity = random.randint(800, 1600)
        payment_selected = random.choice(["upi", "card_emandate"])
    else:  # distraction_or_bounce
        coupon_attempts = 0
        time_on_sec = random.randint(8, 30)
        cart_items = 1
        scroll_depth = random.randint(10, 35)
        exit_velocity = random.randint(900, 2200)
        payment_selected = "none"

    return {
        "event_id": f"ca_{i:04d}",
        "type": "checkout_abandonment",
        "customer": cust,
        "amount": cart_amount,
        "currency": "INR",
        "timestamp": ts.isoformat(),
        "signals": {
            "time_on_checkout_sec": time_on_sec,
            "cart_items": cart_items,
            "cart_amount": cart_amount,
            "coupon_attempts": coupon_attempts,
            "coupon_attempted": coupon_attempts > 0,
            "scroll_depth_pct": scroll_depth,
            "exit_velocity_px_sec": exit_velocity,
            "past_purchases": past_purchases,
            "device": random.choice(DEVICES),
            "payment_method_selected": payment_selected,
            "latent_archetype": intent_archetype,
        },
    }


def gen_subscription_failure(i, base_time=None):
    if base_time is None:
        base_time = datetime.now(IST)
    cust = random.choice(CUSTOMERS)
    ts = base_time - timedelta(hours=random.randint(0, 72), minutes=random.randint(0, 59))
    error_code = pick_weighted(PAYMENT_ERROR_CODES)
    consecutive_fails = random.randint(1, 3)

    return {
        "event_id": f"sf_{i:04d}",
        "type": "subscription_failure",
        "customer": cust,
        "amount": round(random.uniform(299, 4999), 2),
        "currency": "INR",
        "timestamp": ts.isoformat(),
        "signals": {
            "gateway_error_code": error_code,
            "consecutive_failures": consecutive_fails,
            "attempt_number": consecutive_fails,
            "subscription_age_months": random.randint(1, 36),
            "plan": random.choice(["monthly", "annual", "quarterly"]),
            "mandate_type": random.choice(["upi_autopay", "card_emandate"]),
            "card_type": random.choice(CARD_TYPES),
            "customer_tenure_months": cust.get("tenure_months", 12),
            "hour_of_day": ts.hour,
            "day_of_month": ts.day,
        },
    }


def gen_overdue_invoice(i, base_time=None):
    if base_time is None:
        base_time = datetime.now(IST)
    cust = random.choice(CUSTOMERS)
    days_overdue = random.randint(3, 65)
    ts = base_time - timedelta(days=days_overdue)
    has_dispute = random.choice([True, False, False, False])
    prior_promise = random.choice([True, False])

    return {
        "event_id": f"oi_{i:04d}",
        "type": "overdue_invoice",
        "customer": cust,
        "amount": round(random.uniform(8500, 350000), 2),
        "currency": "INR",
        "timestamp": ts.isoformat(),
        "signals": {
            "days_overdue": days_overdue,
            "invoice_count_open": random.randint(1, 4),
            "prior_promise_to_pay": prior_promise,
            "has_dispute": has_dispute,
            "account_tier": random.choice(["smb", "mid_market", "enterprise"]),
            "client_relationship_months": random.randint(3, 48),
        },
    }


def generate_batch(n_per_type=15, seed=42):
    random.seed(seed)
    events = []
    gens = [gen_payment_failure, gen_checkout_abandonment, gen_subscription_failure, gen_overdue_invoice]
    for gen in gens:
        for i in range(n_per_type):
            events.append(gen(i))
    random.shuffle(events)
    return events


if __name__ == "__main__":
    batch = generate_batch(n_per_type=15)
    print(json.dumps(batch, indent=2))
