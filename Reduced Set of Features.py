import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import joblib  # Required to save the PCA model

# Input: Features from previous step
input_path = "folder8_display_features/features.npz"
output_folder = "folder9_reduced_features"
os.makedirs(output_folder, exist_ok=True)

# Load data
data = np.load(input_path, allow_pickle=True)
combined_features = data['combined']
file_names = data['filenames']

# Apply PCA
n_components = 50  # Adjust if needed
pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(combined_features)

# Save PCA features as NPZ
np.savez(os.path.join(output_folder, "pca_features.npz"),
         filenames=file_names,
         reduced=reduced_features)

# Save as CSV
df = pd.DataFrame(reduced_features)
df.insert(0, "filename", file_names)
df.to_csv(os.path.join(output_folder, "pca_features.csv"), index=False)

# ✅ Save the PCA model for later use in GUI
pca_model_path = os.path.join(output_folder, "pca_model.pkl")
joblib.dump(pca, pca_model_path)

print(f"✅ PCA complete: {n_components} components saved in '{output_folder}'")
print(f"✅ PCA model saved to '{pca_model_path}'")
