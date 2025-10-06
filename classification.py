import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load PCA features
data = np.load("folder9_reduced_features/pca_features.npz", allow_pickle=True)
X = data['reduced']
filenames = data['filenames']

# ======== Step 1: Load Actual Labels (Optional) ========
try:
    label_df = pd.read_csv("labels.csv")  # Make sure this file exists
    # Match filenames if needed
    y = label_df['label'].values
    print("✅ Loaded real labels from labels.csv")
except Exception as e:
    print("⚠️ Using dummy labels as fallback.")
    y = np.array([0 if i < len(X) // 2 else 1 for i in range(len(X))])

# ======== Step 2: Split Data ========
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, y, filenames, test_size=0.2, random_state=42
)

# ======== Step 3: Train SVM Model ========
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ======== Step 4: Evaluate ========
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ======== Step 5: Save Everything ========
output_folder = "folder10_classification_model"
os.makedirs(output_folder, exist_ok=True)

# Save model
joblib.dump(clf, os.path.join(output_folder, "model.pkl"))

# Save data splits
pd.DataFrame(X_train).to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(output_folder, "y_test.csv"), index=False)

# Save accuracy and report
with open(os.path.join(output_folder, "accuracy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save predictions
pred_df = pd.DataFrame({
    "filename": filenames_test,
    "actual": y_test,
    "predicted": y_pred
})
pred_df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False)

# ======== Step 6: Done ========
print(f"🎯 Model trained and saved to {output_folder}")
print(f"✅ Accuracy: {accuracy:.4f}")
