import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


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
# 4. Random Forest + Grid Search
# -----------------------------
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

param_grid = {
    'n_estimators': [10, 20, 50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20, 40, 80],
    'min_samples_split': [2, 3, 5, 10, 20, 40]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1', # Evaluated based on f1 score
    n_jobs=-1,
    verbose=1
)

print("\nStarting Grid Search…")
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")


# -----------------------------
# 5. Baseline model
# -----------------------------
dummy = DummyClassifier(strategy='constant', constant=0)
dummy.fit(X_train_scaled, y_train)

y_dummy = dummy.predict(X_test_scaled)

print("\nDummy Classifier Performance:")
print(classification_report(y_test, y_dummy))


# -----------------------------
# 6. Final model evaluation
# -----------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))