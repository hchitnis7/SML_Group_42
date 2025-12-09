import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

df = pd.read_csv("training_data.csv") 
df.head()

# Convert high/low to numeric target
df['target_num'] = df['increase_stock'].map({'high_bike_demand': 1, 'low_bike_demand': 0})

# Add weekend feature (5 = Saturday, 6 = Sunday)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

df[['increase_stock', 'target_num', 'day_of_week', 'is_weekend']].head()

X = df.drop(columns=['increase_stock', 'target_num'])
y = df['target_num']

categorical_features = [
    'hour_of_day',
    'day_of_week',
    'month',
    'holiday',
    'weekday',
    'summertime',
    'is_weekend'
]

categorical_features = [c for c in categorical_features if c in X.columns]

# Convert to categorical values to string
for col in categorical_features:
    X[col] = X[col].astype(str)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

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

model.fit(train_pool, eval_set=test_pool, use_best_model=True)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
model.get_feature_importance(prettified=True)