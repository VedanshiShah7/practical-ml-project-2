import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------- Load Data -------------------

data = pd.read_csv('tPatchGNN/tPatchGNN/train_moratlity.py')  # Make sure train.csv includes 'icustayid' and 'mortality_90d'

# Separate features and target
X = data.drop(columns=['mortality_90d'])
y = data['mortality_90d']

# Store icustayid and drop from training features if present
if 'icustayid' in X.columns:
    X = X.drop(columns=['icustayid'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- Scale Features -------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# ------------------- Build Model -------------------

model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ------------------- Train Model -------------------

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# ------------------- Evaluate -------------------

y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test Accuracy: {acc:.2f}")
print(f"Test AUC: {auc:.2f}")

# ------------------- Save Model as .pkl -------------------

def save_model_as_pkl(model, filename='mortality_dnn_model.pkl'):
    model_json = model.to_json()
    weights = model.get_weights()

    model_dict = {
        'model_json': model_json,
        'weights': weights
    }

    with open(filename, 'wb') as f:
        joblib.dump(model_dict, f)

    print(f"Model saved as '{filename}'")

save_model_as_pkl(model)

# ------------------- Load Model from .pkl -------------------

def load_model_from_pkl(filename='mortality_dnn_model.pkl'):
    with open(filename, 'rb') as f:
        model_dict = joblib.load(f)

    model = tf.keras.models.model_from_json(model_dict['model_json'])
    model.set_weights(model_dict['weights'])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ------------------- Inference Function -------------------

def predict_mortality_dnn(input_csv, model_pkl='mortality_dnn_model.pkl', scaler_pkl='scaler.pkl'):
    # Load the scaler and model
    scaler = joblib.load(scaler_pkl)
    model = load_model_from_pkl(model_pkl)

    # Load input data
    input_data = pd.read_csv(input_csv)

    if 'icustayid' not in input_data.columns:
        raise ValueError("The input CSV must contain an 'icustayid' column.")

    # Extract icustayid and drop it from features
    icustay_ids = input_data['icustayid']
    features = input_data.drop(columns=['icustayid'])

    # Scale input features
    input_scaled = scaler.transform(features)

    # Predict
    prob_preds = model.predict(input_scaled).flatten()
    class_preds = (prob_preds >= 0.5).astype(int)

    # Create result DataFrame
    results = pd.DataFrame({
        'icustayid': icustay_ids,
        'Predicted Class': class_preds,
        'Probability of Survival (0)': np.round((1 - prob_preds) * 100, 2),
        'Probability of Mortality (1)': np.round(prob_preds * 100, 2)
    })

    # Save to CSV
    results.to_csv('predictions_dnn.csv', index=False)
    print("Predictions saved to predictions_dnn.csv")
    return results

# ------------------- Example Inference -------------------

print(predict_mortality_dnn('tPatchGNN/tPatchGNN/train_moratlity.py'))
