from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load all model artifacts
try:
    # Load the trained model
    model = tf.keras.models.load_model('churn_model.h5')
    
    # Load preprocessing objects
    with open('imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('column_transformer.pkl', 'rb') as f:
        column_transformer = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open('preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)
    
    print("All model artifacts loaded successfully!")
    
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    print("Please make sure all .pkl and .h5 files are in the same directory as this Flask app.")

def preprocess_input(data):
    """
    Preprocess input data to match the training data format
    """
    try:
        # Create input array in the exact same order as training data
        # Order: [Credit Score, Geography, Gender, Age, Customer Since, Current Account, Num Products, UPI Enabled, Estimated Yearly Income]
        input_array = np.array([[
            float(data['credit_score']),  # Credit Score
            data['geography'],            # Geography (string)
            data['gender'],              # Gender (string) 
            float(data['age']),          # Age
            float(data['customer_since']), # Customer Since
            float(data['current_account']), # Current Account
            float(data['num_products']),  # Num Products
            float(data['upi_enabled']),   # UPI Enabled
            float(data['estimated_yearly_income'])  # Estimated Yearly Income
        ]], dtype=object)
        
        print(f"Original input array: {input_array}")
        
        # Step 1: Handle missing values in credit score (column 0)
        credit_score_array = np.array([[float(data['credit_score'])]], dtype=float)
        if float(data['credit_score']) == 0:  # Assuming 0 means missing
            credit_score_array = imputer.transform(credit_score_array)
            input_array[0, 0] = credit_score_array[0, 0]
        
        # Step 2: Label encode gender (column 2)
        # The label encoder was fitted on the gender column during training
        encoded_gender = label_encoder.transform([data['gender']])[0]
        input_array[0, 2] = encoded_gender
        
        print(f"After label encoding gender: {input_array}")
        
        # Step 3: Apply column transformer (one-hot encode geography at index 1)
        # The column transformer expects the array in the same format as training
        input_array = column_transformer.transform(input_array)
        
        print(f"After column transformer shape: {input_array.shape}")
        print(f"After column transformer: {input_array}")
        
        # Step 4: Ensure all values are float
        input_array = input_array.astype(float)
        
        # Step 5: Scale the data
        input_array = scaler.transform(input_array)
        
        print(f"Final preprocessed array shape: {input_array.shape}")
        print(f"Final preprocessed array: {input_array}")
        
        return input_array
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        input_data = {
            'credit_score': request.form.get('credit_score', '0'),
            'geography': request.form.get('geography', ''),
            'gender': request.form.get('gender', ''),
            'age': request.form.get('age', '0'),
            'customer_since': request.form.get('customer_since', '0'),
            'current_account': request.form.get('current_account', '0'),
            'num_products': request.form.get('num_products', '0'),
            'upi_enabled': request.form.get('upi_enabled', '0'),
            'estimated_yearly_income': request.form.get('estimated_yearly_income', '0')
        }
        
        print(f"Received input data: {input_data}")
        
        # Validate required fields
        if not input_data['geography'] or not input_data['gender']:
            return render_template('result.html', 
                                 error="Please fill in all required fields.")
        
        # Convert numeric fields
        try:
            input_data['credit_score'] = float(input_data['credit_score'])
            input_data['age'] = float(input_data['age'])
            input_data['customer_since'] = float(input_data['customer_since'])
            input_data['current_account'] = float(input_data['current_account'])
            input_data['num_products'] = float(input_data['num_products'])
            input_data['upi_enabled'] = float(input_data['upi_enabled'])
            input_data['estimated_yearly_income'] = float(input_data['estimated_yearly_income'])
        except ValueError as e:
            return render_template('result.html', 
                                 error=f"Invalid numeric input: {str(e)}")
        
        # Preprocess the input
        processed_input = preprocess_input(input_data)
        
        if processed_input is None:
            return render_template('result.html', 
                                 error="Error in processing input data. Please check your inputs.")
        
        # Make prediction
        prediction_prob = model.predict(processed_input, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        # Prepare result
        result = {
            'prediction': 'Will Close Account' if prediction == 1 else 'Will not Close Account',
            'probability': f"{prediction_prob:.4f}",
            'confidence': f"{max(prediction_prob, 1-prediction_prob):.4f}"
        }
        
        print(f"Prediction result: {result}")
        
        return render_template('result.html', result=result, input_data=input_data)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', 
                             error=f"Error making prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    """
    try:
        data = request.get_json()
        print(f"API received data: {data}")
        
        # Validate required fields
        required_fields = ['credit_score', 'geography', 'gender', 'age', 'customer_since', 
                          'current_account', 'num_products', 'upi_enabled', 'estimated_yearly_income']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                })
        
        # Preprocess the input
        processed_input = preprocess_input(data)
        
        if processed_input is None:
            return jsonify({
                'error': 'Error in processing input data',
                'success': False
            })
        
        # Make prediction
        prediction_prob = model.predict(processed_input, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return jsonify({
            'prediction': prediction,
            'prediction_label': 'Will Close Account' if prediction == 1 else 'Will not Close Account',
            'probability': float(prediction_prob),
            'confidence': float(max(prediction_prob, 1-prediction_prob)),
            'success': True
        })
    
    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/test')
def test_preprocessing():
    """
    Test endpoint to debug preprocessing
    """
    # Test data
    test_data = {
        'credit_score': 600,
        'geography': 'Delhi',
        'gender': 'Male',
        'age': 35,
        'customer_since': 5,
        'current_account': 50000,
        'num_products': 2,
        'upi_enabled': 1,
        'estimated_yearly_income': 500000
    }
    
    processed = preprocess_input(test_data)
    
    return jsonify({
        'original_data': test_data,
        'processed_shape': processed.shape if processed is not None else None,
        'processed_data': processed.tolist() if processed is not None else None,
        'success': processed is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)