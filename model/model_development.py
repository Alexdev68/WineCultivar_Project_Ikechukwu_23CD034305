import os
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# Select 6 required features
selected_features = [
    "alcohol",
    "malic_acid",
    "alcalinity_of_ash",
    "total_phenols",
    "color_intensity",
    "proline"
]

indices = [feature_names.index(f) for f in selected_features]
X = X[:, indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (MANDATORY)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump((model, scaler), "model/wine_cultivar_model.pkl")

# Reload test (NO retraining)
loaded_model, loaded_scaler = joblib.load("model/wine_cultivar_model.pkl")
sample = loaded_scaler.transform([X_test[0]])
print("Reloaded prediction:", loaded_model.predict(sample))
