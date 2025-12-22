import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

train_df = pd.read_csv("catboost_train.csv")
test_df = pd.read_csv("catboost_test.csv")

# Drop the target columns
X_train = train_df.drop(columns=['increase_stock', 'target_num', 'target'])
y_train = train_df['target_num']

X_test = test_df.drop(columns=['increase_stock', 'target_num', 'target'])
y_test = test_df['target_num']

categorical_features = [
    'hour_of_day',
    'day_of_week',
    'month',
    'holiday',
    'weekday',
    'summertime',
    'is_weekend'
]

categorical_features = [c for c in categorical_features if c in X_train.columns]

train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Model training parameters
class_weights = [1, 1.5]
model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    eval_metric='F1',
    class_weights=class_weights,
    random_seed=42,
    verbose=100
)

# Training step
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
model.get_feature_importance(prettified=True)

# Test code
final_test_df = pd.read_csv("catboost_final_test.csv")   # your held-out test file
X_final_test = final_test_df.drop(
    columns=[c for c in ['increase_stock', 'target', 'target_num'] if c in final_test_df.columns]
)

final_test_pool = Pool(X_final_test, cat_features=categorical_features)

# Predict
y_pred_final = model.predict(final_test_pool)
y_prob_final = model.predict_proba(final_test_pool)[:, 1]

# Save predictions
final_test_df["predicted_label"] = y_pred_final
final_test_df["predicted_probability"] = y_prob_final

final_test_df.to_csv("catboost_predictions.csv", index=False)
