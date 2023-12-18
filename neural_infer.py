# Import necessary libraries
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset from the CSV file
# Assuming the last column is the target variable (label)

# Prosodic
dataset = pd.read_csv('./features/emoUERJ_features/aug_mfcc_features.csv')

dataset = dataset.iloc[:,1:]
X = dataset.drop('label', axis=1)  # Features  # Features
# Create a LabelEncoder
label_encoder = LabelEncoder()

# Preprocess Data (example: standardizing numerical features)
scaler = StandardScaler()
X = scaler.fit_transform(X)

test = pd.read_csv('./features/fardo_features/aug_prosodic_features.csv')

X_test = test.iloc[:, 1:-1]  # Features
y_test = label_encoder.fit_transform(test["label"])

X_test = scaler.transform(X_test)

model = keras.models.load_model("best_model.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
