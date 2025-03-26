from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import wandb
import matplotlib.pyplot as plt

def train_model(train_df):
    # Identify the correct mortality column name
    mortality_col = 'mortality_90d_first' if 'mortality_90d_first' in train_df.columns else 'mortality_90d'

    X = train_df.drop(columns=[mortality_col])
    y = train_df[mortality_col]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    # Save model
    joblib.dump(model, 'model/sepsis_model.pkl')

    return model
