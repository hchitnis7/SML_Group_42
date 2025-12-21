import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
import os

DEFAULT_DATADIR = "processed_full"
MODEL_DIR = "models"

FEATURES = [
    "hour_of_day",
    "temp",
    "humidity",
    "windspeed",
    "visibility",
    "dew",
    "precip",
    "snowdepth",
    "cloudcover",
    "is_weekend",
    "is_raining",
    "has_snow",
]

def load_split_csvs(datadir: str):
    train_path = os.path.join(datadir, "rf_train.csv")
    test_path = os.path.join(datadir, "rf_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find rf_train.csv in {datadir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

# Load and prepare the data

train_df, test_df = load_split_csvs(DEFAULT_DATADIR)

X_train = train_df[FEATURES].copy()
y_train = train_df["target_num"]

X_test = test_df[FEATURES].copy()
y_test = test_df["target_num"]

# Random Forest + Grid Search 

rf = RandomForestClassifier(
    random_state=42,
    class_weight=None
)

param_grid = {
    'n_estimators': [50, 100, 200, 500], 
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10, 20] 
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro', # Evaluated based on f1 score
    n_jobs=-1,
    verbose=1
)

print("\nStarting Grid Search...")

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# Final model evaluation
best_model = grid_search.best_estimator_

# Passed X_test directly 
y_pred = best_model.predict(X_test)

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred))

# Baseline Comparison
dummy_clf = DummyClassifier(strategy='constant', constant=0)
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)

acc_dummy = accuracy_score(y_test, y_pred_dummy)
# must specify pos_label=1 to calculate F1 for High Demand
f1_dummy = f1_score(y_test, y_pred_dummy, pos_label=1)

print(f"Baseline (Always Low) Accuracy: {acc_dummy:.4f}")
print(f"Baseline (Always Low) F1-Score: {f1_dummy:.4f}")
