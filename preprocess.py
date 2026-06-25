import pandas as pd
import joblib
import config

def load_preprocessors():
    label_encoders = joblib.load(config.LABEL_ENCODERS_PATH)
    one_hot_encoder = joblib.load(config.ONE_HOT_ENCODER_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    training_columns = joblib.load(config.TRAINING_COLUMNS_PATH)
    return label_encoders, one_hot_encoder, scaler, training_columns

def preprocess_incoming_data(customer_dict):
    try:
        label_encoders, one_hot_encoder, scaler, training_columns = load_preprocessors()
    except Exception as e:
         raise RuntimeError(f"Models not found. Did you run train.py first? {e}")

    df = pd.DataFrame([customer_dict])
    
    # Handle TotalCharges if empty string or missing
    df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', df['MonthlyCharges'] * df.get('tenure', 0)), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Yes/No binary mappings
    binary_cols = config.BINARY_COLS
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)
            
    # Gender mapping
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0}).fillna(0)
        
    # Label Endoding
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by mapping to a default class (first class)
            known_classes = list(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
            df[col] = le.transform(df[col])
            
    # One-Hot Encoding
    one_hot_cols = config.ONE_HOT_COLS
    
    # Add dummy columns for any missing one_hot_cols so transform doesn't fail
    for col in one_hot_cols:
        if col not in df.columns:
            df[col] = "No" # default
            
    encoded_features = one_hot_encoder.transform(df[one_hot_cols])
    encoded_feature_names = one_hot_encoder.get_feature_names_out(one_hot_cols)
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df.drop(columns=one_hot_cols), df_encoded], axis=1)
        
    # Drop customerID if it's there
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # Missing columns handling (to ensure shape matches training)
    for col in training_columns:
        if col not in df.columns:
            if col != 'tenure': # Keep tenure scaling intact if provided
                df[col] = 0
                
    # Scaling
    numerical_cols = config.NUMERICAL_COLS
    # If a value is missing, default to 0
    df[numerical_cols] = df[numerical_cols].fillna(0)
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure column order matches training data exactly
    df = df[training_columns]
    
    return df
