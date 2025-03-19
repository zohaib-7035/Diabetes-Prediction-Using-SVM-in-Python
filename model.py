import numpy as np
import pickle

# Load the trained model
model_path = r'C:/Users/asifn/Downloads/trained_model.sav'  # Use raw string or double backslashes
load_model = pickle.load(open(model_path, 'rb'))

# Making a prediction system
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data to numpy array and reshape
input_array = np.asarray(input_data).reshape(1, -1)

# Make a prediction
prediction_answer = load_model.predict(input_array)  # Use input_array_reshaped

print('The prediction is:', prediction_answer[0])

# Interpret the result
if prediction_answer[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic patient")
