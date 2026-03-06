import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import kagglehub
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
# Assuming the file is in your current directory
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(path, "creditcard.csv"))

print("Dataset Shape:", df.shape)
print("Class Distribution:\n", df['Class'].value_counts(normalize=True))

# 2. Preprocessing: Scale 'Time' and 'Amount'
scaler = StandardScaler()
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original 'Time' and 'Amount'
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Reorder columns to put 'Class' at the end
cols = df.columns.tolist()
cols.remove('Class')
cols.append('Class')
df = df[cols]

# Separate features and target
X_base = df.drop('Class', axis=1)
y = df['Class']

# Initialize Isolation Forest
# We set contamination to the approximate fraud ratio (0.0017)
iso_forest = IsolationForest(n_estimators=100, max_samples='auto',
                             contamination=0.002, random_state=42)

# Fit and predict anomaly scores
print("Training Isolation Forest...")
iso_forest.fit(X_base)

# decision_function yields a score: lower scores indicate more abnormal data
df['Anomaly_Score'] = iso_forest.decision_function(X_base)

# Let's visualize the distribution of anomaly scores for Fraud vs Non-Fraud
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Anomaly_Score', hue='Class', bins=50, kde=True)
plt.title('Isolation Forest Anomaly Scores: Fraud vs. Normal')
plt.xlabel('Anomaly Score (Lower = More Anomalous)')
plt.ylabel('Frequency')
plt.show()

# Update X to include the new Anomaly_Score feature for the Hybrid approach
X_hybrid = df.drop('Class', axis=1)

# We will split using the hybrid feature set. We can drop 'Anomaly_Score' later for the Pure models.
X_train_full, X_test_full, y_train, y_test = train_test_split(X_hybrid, y, test_size=0.2, random_state=42, stratify=y)

# Create Pure feature sets
X_train_pure = X_train_full.drop('Anomaly_Score', axis=1)
X_test_pure = X_test_full.drop('Anomaly_Score', axis=1)

# Apply SMOTE to the Pure Training set
print("Applying SMOTE to Pure dataset...")
smote = SMOTE(random_state=42)
X_train_pure_sm, y_train_pure_sm = smote.fit_resample(X_train_pure, y_train)

# Apply SMOTE to the Hybrid Training set
print("Applying SMOTE to Hybrid dataset...")
X_train_hybrid_sm, y_train_hybrid_sm = smote.fit_resample(X_train_full, y_train)

print(f"Original training shape: {X_train_pure.shape}, Fraud cases: {sum(y_train==1)}")
print(f"SMOTE training shape: {X_train_pure_sm.shape}, Fraud cases: {sum(y_train_pure_sm==1)}")


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, approach_name):
    """Trains a model and returns performance metrics."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {model_name} ({approach_name}) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} ({approach_name}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        'Model': model_name,
        'Approach': approach_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'y_prob': y_prob  # saved for ROC curve plotting
    }


# Initialize Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
}

results = []

# Train and Evaluate Pure Models
for name, model in models.items():
    res = evaluate_model(model, X_train_pure_sm, y_train_pure_sm, X_test_pure, y_test, name, "Pure")
    results.append(res)

# Train and Evaluate Hybrid Models
for name, model in models.items():
    res = evaluate_model(model, X_train_hybrid_sm, y_train_hybrid_sm, X_test_full, y_test, name, "Hybrid")
    results.append(res)

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results).drop('y_prob', axis=1)

print("\n=== Model Comparison Table ===")
print(results_df.pivot(index='Model', columns='Approach', values=['Precision', 'Recall', 'F1-Score', 'ROC-AUC']))

# Plot grouped bar charts for metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pure vs Hybrid Approach Performance Comparison', fontsize=16)

metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    sns.barplot(data=results_df, x='Model', y=metric, hue='Approach', ax=ax, palette='muted')
    ax.set_title(f'{metric} Comparison')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Plot ROC Curves for Hybrid Models
plt.figure(figsize=(10, 8))
for res in results:
    if res['Approach'] == 'Hybrid':
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{res['Model']} (AUC = {res['ROC-AUC']:.4f})")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Hybrid Models)')
plt.legend(loc="lower right")
plt.show()
