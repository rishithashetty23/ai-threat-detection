import pandas as pd
import numpy as np
import os

def load_and_preprocess(file_path):
    """
    Load and preprocess CICIDS-2017 dataset
    """
    print(f"Loading dataset from: {file_path}")
    
    # Load the dataset
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    
    # FIX: Strip whitespace from all column names
    data.columns = data.columns.str.strip()
    
    # Display basic info
    print("Dataset columns (after cleaning):")
    print(data.columns.tolist()[:10])  # Show first 10 columns
    
    # Handle missing values
    print("Handling missing values...")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    
    # Convert labels to binary (BENIGN = 0, any attack = 1)
    print("Converting labels...")
    data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Select features (excluding label column)
    feature_columns = [col for col in data.columns if col != 'Label']
    
    # Select first 20 features for simplicity
    selected_features = feature_columns[:20]
    print(f"Selected features: {selected_features}")
    
    X = data[selected_features]
    y = data['Label']
    
    print(f"Features shape: {X.shape}")
    print(f"Labels distribution: {y.value_counts()}")
    
    return X, y, selected_features

if __name__ == "__main__":
    import os
    
    # Check both dataset/ and dataset/archive/ folders
    possible_paths = ['../dataset', '../dataset/archive']
    
    csv_files = []
    dataset_dir = None
    
    for path in possible_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if files:
                csv_files = files
                dataset_dir = path
                break
    
    if csv_files and dataset_dir:
        print(f"Found {len(csv_files)} CSV files in {dataset_dir}:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        
        # Use the first CSV file found
        dataset_path = os.path.join(dataset_dir, csv_files[0])
        print(f"\nUsing: {dataset_path}")
        
        try:
            X, y, feature_names = load_and_preprocess(dataset_path)
            
            # Save preprocessed data
            X.to_csv('features.csv', index=False)
            y.to_csv('labels.csv', index=False)
            pd.DataFrame(feature_names, columns=['feature_names']).to_csv('feature_names.csv', index=False)
            
            print("\nPreprocessing complete!")
            
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print("No CSV files found!")
        print("Please add your CICIDS-2017 CSV files to:")
        print("- dataset/ folder, or")
        print("- dataset/archive/ folder")
