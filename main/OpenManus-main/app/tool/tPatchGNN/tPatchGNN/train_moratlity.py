import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/train.csv')

# Prepare features and target
X = data.drop(columns=['mortality_90d'])
y = data['mortality_90d']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('clf', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, 'mortality_model.pkl')
print("Model saved as 'mortality_model.pkl'")

# Predict function with binary output
def predict_mortality(input_csv):
    model = joblib.load('mortality_model.pkl')
    print("Model loaded successfully.")

    input_data = pd.read_csv(input_csv)
    class_preds = model.predict(input_data)  # 0 or 1 only
    prob_preds = model.predict_proba(input_data)  # For extra context

    results = []
    for i in range(len(class_preds)):
        results.append({
            'Predicted Mortality': int(class_preds[i]),  # Ensures 0 or 1
            'Probability of Mortality (1)': round(prob_preds[i][1] * 100, 2),
            'Probability of Survival (0)': round(prob_preds[i][0] * 100, 2)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('predictions.csv', index=False)
    return results_df

# Example prediction
print(predict_mortality('/Users/vedanshi/Documents/GitHub/practical-ml-project-2/main/data/test.csv'))
