#!/usr/bin/env python
# coding: utf-8

# In[8]:


import gradio as gr
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.joblib")

# Load the saved model
loaded_model = joblib.load("iris_model.joblib")

# Define the prediction function
def predict_class(sepal_length, sepal_width, petal_length, petal_width):
    # Create an input array from the user's input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Perform prediction using the loaded model
    prediction = loaded_model.predict(input_data)
    
    # Retrieve the class label based on the predicted value
    class_label = iris.target_names[prediction]
    
    return class_label

# Create the Gradio interface
inputs = [
    gr.inputs.Slider(minimum=0, maximum=10, step=0.1, label="Sepal Length"),
    gr.inputs.Slider(minimum=0, maximum=10, step=0.1, label="Sepal Width"),
    gr.inputs.Slider(minimum=0, maximum=10, step=0.1, label="Petal Length"),
    gr.inputs.Slider(minimum=0, maximum=10, step=0.1, label="Petal Width")
]
output = gr.outputs.Textbox(label="Prediction")

interface = gr.Interface(fn=predict_class, inputs=inputs, outputs=output, title="Iris Classification")

# Launch the Gradio interface
interface.launch()


# In[ ]:




