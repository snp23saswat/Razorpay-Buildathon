"""
ml_uplift_optimizer.py
ML Model 3: Dynamic Discount & Net-Uplift Revenue Optimizer
Evaluates candidate incentive actions (No discount, 5% nudge, 10% coupon, 15% VIP code)
and calculates the Expected Net Revenue:
  E[Net] = P(Recover | Action, Features) * Amount * (1 - Discount_rate) - Execution_Cost
Selects the action that strictly MAXIMIZES expected net revenue.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class UpliftOptimizer:
    DISCOUNT_TIERS = [0.0, 0.05, 0.10, 0.15]

    def __init__(self):
        self.model = None
        self.numeric_features = [
            "cart_amount",
            "cart_items",
            "time_on_checkout_sec",
            "coupon_attempts",
            "past_purchases",
            "discount_offered",
        ]
        self.pipeline = None

    def _build_pipeline(self):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42)),
            ]
        )

    def train(self, X_df, y_conversion):
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_df, y_conversion)
        self.model = self.pipeline.named_steps["regressor"]

    def optimize_intervention(self, session_features_dict):
        """
        Evaluates each candidate discount tier and picks the one that maximizes Expected Net Value.
        Returns:
            recommended_discount: float (e.g. 0.05)
            action_name: str
            expected_net: float
            tier_analysis: list of dicts comparing all candidate tiers
            decision_rationale: str
        """
        import pandas as pd

        amount = session_features_dict.get("cart_amount", 1000.0)
        channel_cost = 5.0  # WhatsApp / SMS delivery cost in INR

        tier_results = []
        for d in self.DISCOUNT_TIERS:
            eval_row = dict(session_features_dict)
            eval_row["discount_offered"] = d
            df = pd.DataFrame([eval_row])[self.numeric_features]

            if self.pipeline is not None:
                predicted_prob = float(np.clip(self.pipeline.predict(df)[0], 0.05, 0.95))
            else:
                # Baseline curve
                predicted_prob = min(0.9, 0.2 + d * 3.5)

            # Net revenue calculation if converted
            gross_if_converted = amount * (1.0 - d)
            expected_gross = predicted_prob * gross_if_converted
            expected_cost = channel_cost + (amount * d * predicted_prob)
            expected_net = expected_gross - channel_cost

            tier_results.append({
                "discount_pct": int(d * 100),
                "discount_rate": d,
                "predicted_conversion_prob": round(predicted_prob, 3),
                "expected_gross": round(expected_gross, 2),
                "incentive_cost": round(amount * d, 2),
                "expected_net_revenue": round(expected_net, 2),
            })

        # Pick tier with highest expected net revenue
        best_tier = max(tier_results, key=lambda x: x["expected_net_revenue"])
        best_d = best_tier["discount_rate"]

        if best_d == 0.0:
            action_name = "reminder_message"
            rationale = (
                f"ML Uplift model indicates high organic recovery intent ({best_tier['predicted_conversion_prob']*100:.1f}%). "
                f"Zero discount achieves max net yield (₹{best_tier['expected_net_revenue']:,.2f}) without margin cannibalization."
            )
        else:
            action_name = "targeted_discount_nudge"
            rationale = (
                f"ML Uplift model optimizes for {int(best_d*100)}% discount: lifts conversion from "
                f"{tier_results[0]['predicted_conversion_prob']*100:.1f}% to {best_tier['predicted_conversion_prob']*100:.1f}%, "
                f"maximizing expected net recovery to ₹{best_tier['expected_net_revenue']:,.2f}."
            )

        return {
            "recommended_discount_rate": best_d,
            "recommended_discount_pct": int(best_d * 100),
            "action_name": action_name,
            "expected_net_revenue": best_tier["expected_net_revenue"],
            "tier_analysis": tier_results,
            "rationale": rationale,
        }
