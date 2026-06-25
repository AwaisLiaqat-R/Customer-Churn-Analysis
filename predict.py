import tensorflow as tf
import pickle
import joblib
import config

def load_models():
    ann_model = tf.keras.models.load_model(config.ANN_MODEL_PATH)
    cox_model = None
    try:
        with open(config.COX_MODEL_PATH, 'rb') as f:
            cox_model = pickle.load(f)
    except Exception:
        pass
    return ann_model, cox_model

def get_predictions(processed_df):
    try:
        ann_model, cox_model = load_models()
    except Exception as e:
        raise RuntimeError(f"Failed to load models. Did you run train.py first? {e}")
    
    # ANN prediction (probability of churn)
    ann_prob = ann_model.predict(processed_df, verbose=0)[0][0]
    
    # Cox prediction
    hazard = 0.0
    survival_prob_at_12_months = 0.0
    
    if cox_model:
        try:
            hazard = cox_model.predict_partial_hazard(processed_df).iloc[0]
            survival_prob_at_12_months = cox_model.predict_survival_function(processed_df, times=[12]).iloc[0, 0]
        except Exception:
            pass

    return {
        "churn_probability": float(ann_prob),
        "hazard_score": float(hazard),
        "survival_prob_12m": float(survival_prob_at_12_months)
    }
