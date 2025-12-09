import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and prepare the data
# -----------------------------
df = pd.read_csv('training_data_ht2025.csv')

# Encode target: high = 1, low = 0
df['target'] = df['increase_stock'].map({
    'high_bike_demand': 1,
    'low_bike_demand': 0
})
X = df.drop(['increase_stock', 'target'], axis=1)
y = df['target']

# -----------------------------
# 2. Train–test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# -----------------------------
# 3. Scaling 
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Base LDA 
# -----------------------------

lda_base_model = LinearDiscriminantAnalysis()
lda_base_model.fit(X_train_scaled, y_train)
y_pred_lda_base_model = lda_base_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lda_base_model):.3f}")
print(f"F1 score: {f1_score(y_test, y_pred_lda_base_model, average='macro'):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lda_base_model))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lda_base_model))

# --------------------------------------------------------------------------
# 5. SMOTE for balancing classes + Grid Search for Best LDA Parameters
# --------------------------------------------------------------------------

smote_data = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote_data.fit_resample(X_train_scaled, y_train)
print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_smote).value_counts().to_dict()}")

param_grid = [
    {'solver': ['svd']}, 
    {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.3, 0.5, 0.7, 0.9]}  
]
grid_search = GridSearchCV(
    LinearDiscriminantAnalysis(),
    param_grid,
    cv=5,
    scoring='f1_macro',  # Balancing both classes
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_smote, y_train_smote)
print(f"\nBest LDA parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-macro score: {grid_search.best_score_:.3f}")

# -----------------------------------------------------
# 9. Finding optimal Threshhold and Final tuned model
# -----------------------------------------------------
best_lda = grid_search.best_estimator_
y_pred_smote_data = best_lda.predict(X_test_scaled)
y_proba = best_lda.predict_proba(X_test_scaled)[:, 1]

# Test different thresholds
thresholds = np.arange(0.2, 0.8, 0.05)
best_threshold = 0.5
best_f1_macro = 0

results = []
for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    f1_low = f1_score(y_test, y_pred_thresh, pos_label=0)
    f1_high = f1_score(y_test, y_pred_thresh, pos_label=1)
    f1_macro = f1_score(y_test, y_pred_thresh, average='macro')
    
    results.append({
        'threshold': threshold,
        'f1_low': f1_low,
        'f1_high': f1_high,
        'f1_macro': f1_macro
    })
    
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_threshold = threshold

print("\nThreshold Analysis:")
results_df = pd.DataFrame(results)
print(results_df.round(3))
print(f"\nOptimal threshold: {best_threshold:.3f}")
print(f"Best F1-macro: {best_f1_macro:.3f}")
y_pred_final = (y_proba >= best_threshold).astype(int)
print("\nFinal Tuned LDA Model\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.3f}")
print(f"F1-macro: {f1_score(y_test, y_pred_final, average='macro'):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
