import os

# Directories
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Data files
DATA_PATH = os.path.join(DATA_DIR, 'Telco-Customer-Churn.csv')
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Model file paths
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, 'label_encoders.pkl')
ONE_HOT_ENCODER_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
TRAINING_COLUMNS_PATH = os.path.join(MODELS_DIR, 'training_columns.pkl')
ANN_MODEL_PATH = os.path.join(MODELS_DIR, 'churn_ann_model.h5')
COX_MODEL_PATH = os.path.join(MODELS_DIR, 'cox_model.pkl')

# Feature definitions
BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
LABEL_ENCODE_COLS = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']
ONE_HOT_COLS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TARGET_COL = 'Churn'

# Training Hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
ANN_EPOCHS = 20
ANN_BATCH_SIZE = 32
ANN_VALIDATION_SPLIT = 0.2
