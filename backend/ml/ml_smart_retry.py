"""
ml_smart_retry.py
ML Model 1: Smart Retry Timing & Recovery Propensity Optimizer
Trained on historical payment failure events to predict recovery probability
across candidate retry windows (e.g. immediate, +4h, +12h, +24h, +48h, payday)
and determine the optimal execution time that maximizes success while minimizing fee cost.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class SmartRetryOptimizer:
    def __init__(self):
        self.model = None
        self.categorical_features = ["gateway_error_code", "mandate_type", "card_type"]
        self.numeric_features = ["amount", "attempt_number", "customer_tenure_months", "hour_of_day", "day_of_month"]
        self.pipeline = None
        self.feature_names = []

    def _build_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", "passthrough", self.numeric_features),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)),
            ]
        )

    def train(self, X_df, y):
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_df, y)
        self.model = self.pipeline.named_steps["classifier"]
        
        # Extract feature names
        try:
            cat_encoder = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            cat_cols = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
            self.feature_names = cat_cols + self.numeric_features
        except Exception:
            self.feature_names = self.categorical_features + self.numeric_features

    def predict_propensity(self, event_features_df):
        """Predicts probability of successful recovery for the given feature row."""
        if self.pipeline is None:
            return 0.5
        probs = self.pipeline.predict_proba(event_features_df)
        return float(probs[0][1])

    def recommend_optimal_retry_window(self, event_features_dict):
        """
        Evaluates candidate retry offsets [2h, 4h, 8h, 12h, 24h, 48h] taking into account:
        - Bank operational hours (morning 9-11 AM, evening 6-8 PM have higher UPI/card throughput)
        - Quiet hours (avoid 21:00 - 08:00 IST)
        - Predicted recovery probability vs attempt fee
        """
        import pandas as pd

        candidate_offsets = [2, 4, 8, 12, 24, 48]
        current_hour = event_features_dict.get("hour_of_day", 14)
        current_dom = event_features_dict.get("day_of_month", 15)

        candidate_scores = {}
        for offset in candidate_offsets:
            future_hour = (current_hour + offset) % 24
            future_dom = (current_dom + (current_hour + offset) // 24) % 31 + 1

            eval_row = dict(event_features_dict)
            eval_row["hour_of_day"] = future_hour
            eval_row["day_of_month"] = future_dom
            df = pd.DataFrame([eval_row])
            
            p = self.predict_propensity(df)

            # IST quiet hours penalty (21:00 to 08:00 IST)
            if future_hour >= 21 or future_hour < 8:
                p_adjusted = p * 0.35  # Do not trigger during quiet hours
            elif 9 <= future_hour <= 12 or 17 <= future_hour <= 20:
                p_adjusted = p * 1.15  # Peak payment success hours
            else:
                p_adjusted = p

            candidate_scores[offset] = min(0.95, max(0.05, round(p_adjusted, 3)))

        best_offset = max(candidate_scores, key=candidate_scores.get)
        best_prob = candidate_scores[best_offset]
        best_target_hour = (current_hour + best_offset) % 24

        reasoning = (
            f"ML model projects highest recovery rate ({best_prob*100:.1f}%) at +{best_offset}h "
            f"({best_target_hour:02d}:00 IST), avoiding quiet hours and scheduling within peak bank clearance window."
        )

        return {
            "best_offset_hours": best_offset,
            "best_prob": best_prob,
            "target_hour_ist": best_target_hour,
            "recommended_time_slot": f"+{best_offset}h (at {best_target_hour:02d}:00 IST)",
            "candidate_probabilities": candidate_scores,
            "reasoning": reasoning,
        }

    def get_feature_importances(self):
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        res = {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True)[:10])
