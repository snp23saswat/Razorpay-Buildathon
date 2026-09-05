"""
ml_intent_classifier.py
ML Model 2: Checkout Abandonment Intent & Behavioral Classifier
Classifies abandoned checkout sessions into distinct root causes:
- price_sensitivity
- friction_hesitation
- technical_issue
- distraction_or_bounce
Uses behavioral telemetry (cart size, coupon attempts, dwell time, exit velocity, scroll depth).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class AbandonmentIntentClassifier:
    CLASSES = ["price_sensitivity", "friction_hesitation", "technical_issue", "distraction_or_bounce"]

    def __init__(self):
        self.numeric_features = [
            "time_on_checkout_sec",
            "cart_items",
            "cart_amount",
            "coupon_attempts",
            "scroll_depth_pct",
            "exit_velocity_px_sec",
            "past_purchases",
        ]
        self.categorical_features = ["device", "payment_method_selected"]
        self.pipeline = None
        self.feature_names = []

    def _build_pipeline(self):
        from sklearn.preprocessing import OneHotEncoder
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", StandardScaler(), self.numeric_features),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(n_estimators=70, max_depth=5, random_state=42)),
            ]
        )

    def train(self, X_df, y):
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_df, y)

        try:
            cat_encoder = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            cat_cols = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
            self.feature_names = cat_cols + self.numeric_features
        except Exception:
            self.feature_names = self.categorical_features + self.numeric_features

    def predict_intent(self, event_features_df):
        """
        Returns:
            primary_intent: str
            confidence: float
            probabilities: dict of {class_name: prob}
            top_factors: list of contributing signals
        """
        import pandas as pd

        if self.pipeline is None:
            return {
                "intent": "friction_hesitation",
                "confidence": 0.6,
                "probabilities": {c: 0.25 for c in self.CLASSES},
                "top_factors": ["Heuristic default"],
            }

        probs = self.pipeline.predict_proba(event_features_df)[0]
        classes = self.pipeline.classes_
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        best_idx = np.argmax(probs)
        best_class = str(classes[best_idx])
        best_conf = round(float(probs[best_idx]), 4)

        # Build explainability factors
        row = event_features_df.iloc[0]
        factors = []
        if row.get("coupon_attempts", 0) > 0:
            factors.append(f"Tried {row.get('coupon_attempts')} coupon code(s)")
        if row.get("time_on_checkout_sec", 0) > 180:
            factors.append("Extended checkout dwell time (>3 min)")
        elif row.get("time_on_checkout_sec", 0) < 25:
            factors.append("Immediate bounce (<25s)")
        if row.get("exit_velocity_px_sec", 0) > 1200:
            factors.append("High exit mouse velocity")
        if row.get("cart_items", 0) >= 3:
            factors.append(f"Multi-item cart ({row.get('cart_items')} items)")

        if not factors:
            factors.append("Standard checkout behavior pattern")

        return {
            "intent": best_class,
            "confidence": best_conf,
            "probabilities": prob_dict,
            "top_factors": factors,
        }

    def get_feature_importances(self):
        if self.pipeline is None:
            return {}
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            return {}
        importances = classifier.feature_importances_
        res = {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True)[:10])
