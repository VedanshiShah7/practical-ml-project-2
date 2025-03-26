from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

def train_model(train_df):
    mortality_col = 'mortality_90d' if 'mortality_90d' in train_df.columns else 'mortality_90d_first'
    X = train_df.drop(columns=[mortality_col])
    y = train_df[mortality_col]

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    print(f"Validation F1 Score: {f1_score(y_val, y_pred):.4f}")
    
    joblib.dump(model, "model/sepsis_model.pkl")

# Run training
train_df = pd.read_csv("data/train.csv")
train_model(train_df)
