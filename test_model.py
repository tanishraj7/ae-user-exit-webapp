import numpy as np
import pickle
import tensorflow as tf

def test_model_artifacts():
    """
    Test if all model artifacts can be loaded and work correctly
    """
    try:
        print("Testing model artifacts...")
        
        # Test loading all artifacts
        print("1. Loading model...")
        model = tf.keras.models.load_model('churn_model.h5')
        print(f"   Model loaded successfully. Input shape: {model.input_shape}")
        
        print("2. Loading imputer...")
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        print("   Imputer loaded successfully")
        
        print("3. Loading label encoder...")
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"   Label encoder loaded. Classes: {label_encoder.classes_}")
        
        print("4. Loading column transformer...")
        with open('column_transformer.pkl', 'rb') as f:
            column_transformer = pickle.load(f)
        print("   Column transformer loaded successfully")
        
        print("5. Loading scaler...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("   Scaler loaded successfully")
        
        # Test preprocessing pipeline
        print("\n6. Testing preprocessing pipeline...")
        test_data = {
            'credit_score': 600.0,
            'geography': 'Delhi',
            'gender': 'Male',
            'age': 35.0,
            'customer_since': 5.0,
            'current_account': 50000.0,
            'num_products': 2.0,
            'upi_enabled': 1.0,
            'estimated_yearly_income': 500000.0
        }
        
        # Step-by-step preprocessing
        print("   Creating input array...")
        input_array = np.array([[
            test_data['credit_score'],
            test_data['geography'],
            test_data['gender'],
            test_data['age'],
            test_data['customer_since'],
            test_data['current_account'],
            test_data['num_products'],
            test_data['upi_enabled'],
            test_data['estimated_yearly_income']
        ]], dtype=object)
        print(f"   Input array shape: {input_array.shape}")
        print(f"   Input array: {input_array}")
        
        print("   Applying label encoding to gender...")
        input_array[0, 2] = label_encoder.transform([test_data['gender']])[0]
        print(f"   After label encoding: {input_array}")
        
        print("   Applying column transformer...")
        transformed = column_transformer.transform(input_array)
        print(f"   After column transformer shape: {transformed.shape}")
        print(f"   After column transformer: {transformed}")
        
        print("   Converting to float...")
        transformed = transformed.astype(float)
        print(f"   After converting to float: {transformed}")
        
        print("   Applying scaler...")
        scaled = scaler.transform(transformed)
        print(f"   After scaling shape: {scaled.shape}")
        print(f"   After scaling: {scaled}")
        
        print("   Making prediction...")
        prediction = model.predict(scaled, verbose=0)
        print(f"   Prediction: {prediction[0][0]}")
        print(f"   Binary prediction: {'Will Churn' if prediction[0][0] > 0.5 else 'Will Not Churn'}")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_inputs():
    """
    Test with different input combinations
    """
    test_cases = [
        {
            'name': 'High churn risk',
            'data': {
                'credit_score': 400.0,
                'geography': 'Mumbai',
                'gender': 'Female',
                'age': 65.0,
                'customer_since': 1.0,
                'current_account': 0.0,
                'num_products': 1.0,
                'upi_enabled': 0.0,
                'estimated_yearly_income': 100000.0
            }
        },
        {
            'name': 'Low churn risk',
            'data': {
                'credit_score': 800.0,
                'geography': 'Delhi',
                'gender': 'Male',
                'age': 35.0,
                'customer_since': 8.0,
                'current_account': 100000.0,
                'num_products': 4.0,
                'upi_enabled': 1.0,
                'estimated_yearly_income': 800000.0
            }
        }
    ]
    
    try:
        # Load artifacts
        model = tf.keras.models.load_model('churn_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('column_transformer.pkl', 'rb') as f:
            column_transformer = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("\nTesting different input scenarios:")
        print("="*50)
        
        for test_case in test_cases:
            print(f"\nTest case: {test_case['name']}")
            data = test_case['data']
            
            # Preprocess
            input_array = np.array([[
                data['credit_score'], data['geography'], data['gender'],
                data['age'], data['customer_since'], data['current_account'],
                data['num_products'], data['upi_enabled'], data['estimated_yearly_income']
            ]], dtype=object)
            
            input_array[0, 2] = label_encoder.transform([data['gender']])[0]
            transformed = column_transformer.transform(input_array)
            scaled = scaler.transform(transformed.astype(float))
            
            # Predict
            prediction = model.predict(scaled, verbose=0)[0][0]
            result = 'Will Close Account' if prediction > 0.5 else 'Will Not Close Account'
            
            print(f"  Prediction probability: {prediction:.4f}")
            print(f"  Result: {result}")
            
    except Exception as e:
        print(f"Error in testing different inputs: {e}")

if __name__ == "__main__":
    print("Testing Model Artifacts")
    print("="*30)
    
    success = test_model_artifacts()
    
    if success:
        test_different_inputs()
    else:
        print("\nPlease fix the issues above before running the Flask app.")