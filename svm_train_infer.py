# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # For classification; use SVR for regression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset from the CSV file
# Assuming the last column is the target variable (label)

# Prosodic
dataset = pd.read_csv('./features/emoUERJ_features/aug_prosodic_features.csv')

selected_features = ["band_energy_difference", "band_density_difference", "f1_median", "min_intensity", "fitch_vtl", "f1_mean", "mff", "q3_pitch", "mean_pitch",
"median_intensity", "stddev_hnr", "stddev_pitch", "max_pitch", "q1_pitch", "center_of_gravity_spectrum", "formant_dispersion", "max_gne",
"stddev_intensity", "stddev_spectrum", "stddev_gne", "central_moment_spectrum", "mean_gne", "sum_gne", "f2_mean", "mean_absolute_pitch_slope",
"relative_max_intensity_time", "voiced_fraction", "relative_min_hnr_time"] # 28 best features
X_train = dataset.iloc[:, 1:-1]  # Features  # Features
y_train = dataset.loc[:, ["label"]]   # Target variable

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

test = pd.read_csv('./features/fardo_features/aug_prosodic_features.csv')

X_test = test.iloc[:, 1:-1]  # Features
y_test = test.loc[:, ["label"]]   # Target variable

X_test = scaler.transform(X_test)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

print(y_test)
print(y_pred)

# Evaluate the performance (for classification)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional metrics (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred))
