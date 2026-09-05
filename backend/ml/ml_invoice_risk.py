"""
ml_invoice_risk.py
ML Model 4: B2B Invoice Default Risk & Expected DSO Delay Regressor
Evaluates B2B account characteristics and overdue signals to predict:
1. Default / Write-off Risk Score (0.0 to 1.0)
2. Expected Days-to-Pay delay
3. Optimal Dunning Tone (Soft reminder, Structured schedule, Payment plan, Executive escalation)
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class InvoiceRiskModel:
    RISK_TIERS = ["low_risk", "medium_risk", "high_default_risk"]

    def __init__(self):
        self.categorical_features = ["account_tier", "has_dispute", "prior_promise_to_pay"]
        self.numeric_features = ["amount", "days_overdue", "invoice_count_open", "client_relationship_months"]
        self.classifier_pipeline = None
        self.regressor_pipeline = None

    def _build_pipelines(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", StandardScaler(), self.numeric_features),
            ]
        )
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=70, max_depth=6, random_state=42)),
            ]
        )
        reg = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", GradientBoostingRegressor(n_estimators=60, max_depth=5, random_state=42)),
            ]
        )
        return clf, reg

    def train(self, X_df, y_risk_class, y_days_to_pay):
        self.classifier_pipeline, self.regressor_pipeline = self._build_pipelines()
        self.classifier_pipeline.fit(X_df, y_risk_class)
        self.regressor_pipeline.fit(X_df, y_days_to_pay)

    def assess_invoice(self, invoice_features_dict):
        """
        Assesses invoice risk and returns recommended intervention strategy.
        """
        import pandas as pd

        df = pd.DataFrame([invoice_features_dict])

        if self.classifier_pipeline is not None and self.regressor_pipeline is not None:
            risk_probs = self.classifier_pipeline.predict_proba(df)[0]
            classes = self.classifier_pipeline.classes_
            prob_map = {c: round(float(p), 3) for c, p in zip(classes, risk_probs)}
            predicted_risk = str(self.classifier_pipeline.predict(df)[0])
            expected_days_to_pay = max(1, round(float(self.regressor_pipeline.predict(df)[0]), 1))
            default_prob = prob_map.get("high_default_risk", 0.15)
        else:
            # Heuristic fallback
            days = invoice_features_dict.get("days_overdue", 15)
            default_prob = min(0.9, days / 75.0)
            predicted_risk = "high_default_risk" if default_prob > 0.5 else "medium_risk" if default_prob > 0.25 else "low_risk"
            expected_days_to_pay = days + 10
            prob_map = {"low_risk": 0.5, "medium_risk": 0.3, "high_default_risk": 0.2}

        # Strategy selection
        if predicted_risk == "high_default_risk" or default_prob > 0.55:
            strategy = "executive_human_escalation"
            tone = "Urgent / Settlement Offer"
            confidence = 0.88
            action_ladder = [("formal_notice", 0.4), ("human_escalation", 0.45)]
        elif predicted_risk == "medium_risk":
            strategy = "structured_payment_plan"
            tone = "Professional Firm Notice"
            confidence = 0.78
            action_ladder = [("promise_to_pay_followup", 0.55), ("formal_notice", 0.35), ("human_escalation", 0.2)]
        else:
            strategy = "courtesy_dunning"
            tone = "Gentle Account Statement"
            confidence = 0.85
            action_ladder = [("gentle_reminder", 0.65), ("formal_notice", 0.3), ("human_escalation", 0.15)]

        return {
            "risk_tier": predicted_risk,
            "default_probability": round(default_prob, 3),
            "expected_additional_days": expected_days_to_pay,
            "recommended_strategy": strategy,
            "recommended_tone": tone,
            "confidence": confidence,
            "action_ladder": action_ladder,
            "probabilities": prob_map,
        }
