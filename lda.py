import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

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
# 4. LDA Base Model
# -----------------------------
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled,y_train)
lda_predictions = lda_model.predict(X_test_scaled)
print ("LDA Prediction Results ")
print (" Accuracy Score :", accuracy_score (y_test , lda_predictions) )
print (" Precision Score :", precision_score (y_test , lda_predictions) )
print (" Recall Score :", recall_score (y_test , lda_predictions) )
print (" F1 Score :", f1_score (y_test , lda_predictions) )
print ("\nClassification Report :\n", classification_report (y_test ,lda_predictions) )
print("\nConfusion matrix:/n", confusion_matrix(y_test, lda_predictions))

