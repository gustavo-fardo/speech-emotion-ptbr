# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC  # For classification; use SVR for regression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset from the CSV file
# Assuming the last column is the target variable (label)

ext = ['praat', 'mfcc', 'melspec', 'pyaudioanalysis', 'aug_praat', 'aug_mfcc', 'aug_melspec', 'aug_pyaudioanalysis', 'aug1_praat', 'aug1_mfcc', 'aug1_melspec', 'aug1_pyaudioanalysis']
for e in ext:    
    print(e)
    # Prosodic
    dataset = pd.read_csv(f'./features/emoUERJ_features/{e}_features.csv')

    # selected_features = ["band_energy_difference", "band_density_difference", "f1_median", "min_intensity", "fitch_vtl", "f1_mean", "mff", "q3_pitch", "mean_pitch",
    # "median_intensity", "stddev_hnr", "stddev_pitch", "max_pitch", "q1_pitch", "center_of_gravity_spectrum", "formant_dispersion", "max_gne",
    # "stddev_intensity", "stddev_spectrum", "stddev_gne", "central_moment_spectrum", "mean_gne", "sum_gne", "f2_mean", "mean_absolute_pitch_slope",
    # "relative_max_intensity_time", "voiced_fraction", "relative_min_hnr_time"] # 28 best features
    # X = dataset.loc[:, selected_features]  # Features
    X = dataset.drop(['label', 'sound_filepath'], axis=1, errors='ignore') # Features
    y = dataset.loc[:, ["label"]]  # Target variable

    # Create a LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the "label" column
    y_cross = label_encoder.fit_transform(dataset["label"])

    # TSFEL
    # dataset = pd.read_csv('./emoUERJ_features/tsfel_features.csv')
    # X = dataset.iloc[:, :-2]  # Features
    # y = dataset.iloc[:, -1]   # Target variable

    # pyAudioAnalysis
    # dataset = pd.read_csv('./emoUERJ_features/pyaudioanalysis_features.csv')
    # X = dataset.iloc[:, :-2]  # Features
    # y = dataset.iloc[:, -1]   # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (important for SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create an SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0)#, random_state=42)

    # Train the SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the performance (for classification)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # # Additional metrics (optional)
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))

    # Perform cross-validation with 5 folds
    cv_scores = cross_val_score(svm_classifier, X, y_cross, cv=5, scoring='f1_weighted')

    # Display the cross-validation scores
    # print("Cross-Validation Scores:", cv_scores)
    print("Mean F1:", cv_scores.mean())
