from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger


app=Flask(__name__)


Swagger(app)
pickle_in=open('BankNote_Authentication.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome all'


@app.route('/predict',methods=['GET'])
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: integer
        required: true
      - name: skewness
        in: query
        type: integer
        required: true
      - name: kurtosis
        in: query
        type: integer
        required: true      
      - name: entropy
        in: query
        type: integer
        required: true
    responses:
        200:
            description: The output values

    """

    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    kurtosis=request.args.get('kurtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,kurtosis,entropy]])
    return "The predicted value is"+str(prediction)   


@app.route('/predict_file',methods=['POST'])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
       - name: file
         in: formData
         type: file
         required: true
       
    responses:
        200:
            description: The output values
            
    """
    
    df_file=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_file)
    
    return str(list(prediction))
    

if __name__=='__main__':
    app.run(port=5000)