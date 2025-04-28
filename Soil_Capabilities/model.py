import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('dataset1.csv')
df_copy = df.copy()

# Separate features and labels
labels = df_copy['Output']
features = df_copy.drop('Output', axis=1)

# Apply log10 transformation to numerical features
transformedFeatures = features.apply(lambda x: np.log10(x + 1e-10) if np.issubdtype(x.dtype, np.number) else x)

trainInput, validationInput, trainTarget, validationTarget = train_test_split(
    transformedFeatures, labels, test_size=0.2, shuffle=True, random_state=42
)

print("Train Data Shape:", trainInput.shape)

randomForestModel = RandomForestClassifier(
    criterion='gini',
    max_depth=10,
    max_features='sqrt',
    n_estimators=300,
    random_state=42
)
randomForestModel.fit(trainInput, trainTarget.values.ravel())

joblib.dump(randomForestModel, 'random_forest_model.joblib')
print("Model saved as random_forest_model.joblib")