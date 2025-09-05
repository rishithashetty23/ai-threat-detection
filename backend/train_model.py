import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

def train_model():
    """
    Train the AI model for threat detection
    """
    print("Loading preprocessed data...")
    
    try:
        X = pd.read_csv('features.csv')
        y = pd.read_csv('labels.csv').values.ravel()
        feature_names = pd.read_csv('feature_names.csv')['feature_names'].tolist()
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save the model
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        joblib.dump(model_data, 'cicids_model.pkl')
        
        print("\nModel saved as 'cicids_model.pkl'")
        print("Training complete!")
        
    except FileNotFoundError:
        print("Error: Preprocessed data files not found.")
        print("Please run preprocess.py first.")

if __name__ == "__main__":
    train_model()
