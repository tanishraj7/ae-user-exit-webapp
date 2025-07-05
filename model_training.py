import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def create_and_save_model_artifacts(data_file_path):
    """
    Creates and saves all necessary model artifacts for Flask deployment
    """
    
    # Load data
    print("Loading data...")
    data = pd.read_excel(data_file_path) if data_file_path.endswith('.xlsx') else pd.read_csv(data_file_path)
    
    # Prepare features and target
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Handle missing values in Credit Score column (index 0)
    print("Handling missing values...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Convert to float for imputation
    credit_scores = x[:, 0:1].astype(float)
    imputer.fit(credit_scores)
    x[:, 0:1] = imputer.transform(credit_scores)
    
    # Save imputer
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    print("Saved imputer.pkl")
    
    # Label encoding for Gender (assuming it's at index 2)
    print("Label encoding gender...")
    le = LabelEncoder()
    x[:, 2] = le.fit_transform(x[:, 2])
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Saved label_encoder.pkl")
    print(f"Gender classes: {le.classes_}")
    
    # One hot encoding for Geography (assuming it's at index 1)
    print("One-hot encoding geography...") 
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [1])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    
    # Save column transformer
    with open('column_transformer.pkl', 'wb') as f:
        pickle.dump(ct, f)
    print("Saved column_transformer.pkl")
    print(f"Feature names after transformation: {ct.get_feature_names_out()}")
    print(f"Shape after column transformation: {x.shape}")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler.pkl")
    
    # Reshape y_train
    y_train = y_train.reshape(-1, 1)
    
    # Create and train neural network
    print("Creating and training neural network...")
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=5, activation='relu', input_dim=x_train.shape[1]))
    ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = ann.fit(x_train, y_train, batch_size=32, epochs=120, verbose=1)
    
    # Save the trained model
    ann.save('churn_model.h5')
    print("Saved churn_model.h5")
    
    # Make predictions and evaluate
    y_pred = ann.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Print evaluation metrics
    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save feature names for reference
    feature_names = [
        'CreditScore', 'Geography_Bengaluru', 'Geography_Delhi', 'Geography_Mumbai', 
        'Gender', 'Age', 'CustomerSince', 'CurrentAccount', 'NumProducts', 
        'UPIEnabled', 'EstimatedYearlyIncome'
    ]
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Saved feature_names.pkl")
    
    # Save column info for preprocessing
    preprocessing_info = {
        'credit_score_col': 0,
        'geography_col': 1,
        'gender_col': 2,
        'feature_order': [
            'Credit Score', 'Geography', 'Gender', 'Age', 'Customer Since',
            'Current Account', 'Num of products', 'UPI Enabled', 'Estimated Yearly Income'
        ]
    }
    
    with open('preprocessing_info.pkl', 'wb') as f:
        pickle.dump(preprocessing_info, f)
    print("Saved preprocessing_info.pkl")
    
    print("\nAll model artifacts saved successfully!")
    print("Files created:")
    print("- imputer.pkl")
    print("- label_encoder.pkl") 
    print("- column_transformer.pkl")
    print("- scaler.pkl")
    print("- churn_model.h5")
    print("- feature_names.pkl")
    print("- preprocessing_info.pkl")
    
    return ann, history

if __name__ == "__main__":
    # Replace 'your_data_file.xlsx' with your actual data file path
    data_file_path = 'ae_user_exit.xlsx'  # Change this to your file path
    
    try:
        model, training_history = create_and_save_model_artifacts(data_file_path)
        print("\nModel training and artifact creation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure your data file path is correct and the file exists.")