import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
import joblib

df = pd.read_csv('dataset1.csv') # Load the dataset
df_copy = df.copy() # Make a copy of the DataFrame for manipulation and editing without affecting the original dataset

labels = df_copy['Output'] # Separate features and labels
features = df_copy.drop('Output', axis=1) # Drop the 'Output' column from the DataFrame

transformedFeatures = features.apply(lambda x: np.log10(x + 1e-10) if np.issubdtype(x.dtype, np.number) else x) # Apply log transformation to numerical features for better distribution

trainInput, validationInput, trainTarget, validationTarget = train_test_split(
    transformedFeatures, labels, test_size=0.2, shuffle=True, random_state=42 # Split the dataset into training 80% and validation 20% sets, 42 for reproducibility
)

print("Train Data Shape:", trainInput.shape) # Print the number of rows and columns in the training data

randomForestModel = RandomForestClassifier(
    criterion='gini', # Use Gini impurity for splitting
    max_depth=10, # Maximum depth of the tree
    max_features='sqrt', # Use square root of the number of features for splitting
    n_estimators=300, # Number of trees in the forest
    random_state=42 # Random state for reproducibility
)
randomForestModel.fit(trainInput, trainTarget.values.ravel()) # Train the model on the training data

joblib.dump(randomForestModel, 'random_forest_model.joblib') # Save the trained model to a file for later use
print("Model saved as random_forest_model.joblib") # Print a message indicating that the model has been saved