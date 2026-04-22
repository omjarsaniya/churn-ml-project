import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from preprocess import load_data, clean_data, split_data
from pipeline import create_pipeline

# Load data
df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean
df = clean_data(df)

# Split features + target
X, y = split_data(df)

# Train-test split (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = create_pipeline(X)

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))

# Threshold tuning
y_custom = (y_prob > 0.3).astype(int)

print("\nCustom Threshold Report:")
print(classification_report(y_test, y_custom))

# Save model
joblib.dump(pipeline, "../models/pipeline.pkl")

print("\nPipeline trained and saved!")