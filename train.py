import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
import tensorflow as tf
import joblib
import pickle
import os
import config

def main():
    print("Initializing training script...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Load dataset
    if not os.path.exists(config.DATA_PATH):
        print("Dataset not found locally. Downloading from IBM repo...")
        df = pd.read_csv(config.DATA_URL)
        df.to_csv(config.DATA_PATH, index=False)
        print("Download complete.")
    else:
        df = pd.read_csv(config.DATA_PATH)
        print("Dataset loaded from local.")

    print(f"Dataset shape: {df.shape}")

    # Handling missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Yes/No columns binary encoding
    binary_cols = config.BINARY_COLS + [config.TARGET_COL]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Label Encoding for specific categorical columns
    label_encoders = {}
    for col in config.LABEL_ENCODE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    joblib.dump(label_encoders, config.LABEL_ENCODERS_PATH)

    # Gender binary encoding
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    # One-Hot Encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = one_hot_encoder.fit_transform(df[config.ONE_HOT_COLS])
    encoded_feature_names = one_hot_encoder.get_feature_names_out(config.ONE_HOT_COLS)
    joblib.dump(one_hot_encoder, config.ONE_HOT_ENCODER_PATH)

    # Combine back into DataFrame
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df.drop(columns=config.ONE_HOT_COLS), df_encoded], axis=1)

    # Drop customerID
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Scaling numerical columns
    scaler = StandardScaler()
    df[config.NUMERICAL_COLS] = scaler.fit_transform(df[config.NUMERICAL_COLS])
    joblib.dump(scaler, config.SCALER_PATH)

    print("Preprocessing components saved.")

    # Survival analysis targets
    df['time'] = df['tenure']
    df['event'] = df[config.TARGET_COL]
    
    # Define features and targets
    X = df.drop(columns=['time', 'event', 'tenure', config.TARGET_COL])
    y = df[['time', 'event']]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    
    # Save training columns list for inference
    joblib.dump(list(x_train.columns), config.TRAINING_COLUMNS_PATH)

    # Build and Train ANN Model
    print("Training ANN Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train['event'], epochs=config.ANN_EPOCHS, validation_split=config.ANN_VALIDATION_SPLIT, batch_size=config.ANN_BATCH_SIZE, verbose=1)

    model.save(config.ANN_MODEL_PATH)
    print(f"ANN model saved ({config.ANN_MODEL_PATH})")

    # Build and Train Cox PH Model
    print("Training Cox PH Model...")
    cox_df = pd.concat([x_train, y_train], axis=1)
    
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col='time', event_col='event', show_progress=True)
        with open(config.COX_MODEL_PATH, 'wb') as f:
            pickle.dump(cph, f)
        print(f"Cox PH Model saved ({config.COX_MODEL_PATH})")
    except Exception as e:
        print(f"Warning: Cox PH Model failed to train. Skipping Cox model. Error: {e}")

    print("Training complete!")

if __name__ == "__main__":
    main()
