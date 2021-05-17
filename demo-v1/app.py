import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from test import return_test_sample
from multi_train import train_fcn
import os 
import logging as logger
import boto3
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# from gevent.pywsgi import WSGIServer
logger.basicConfig(level='DEBUG')

# Creating the app
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

# Loading the model from s3 bucket
s3 = boto3.client('s3') #low-level functional API
response = s3.get_object(Bucket='<bucket>', Key='<model-path>')
body = response['Body'].read()
model = pickle.loads(body)

# model = pickle.load(open('./models/LM_model.pkl'))

# LOCAL ACCESS
# Loading the best model after running the multi_train
# files = os.listdir('./mlruns/0/')
# r2_values = []
# file_names = []
# for dir_id in files:
#     if dir_id != 'meta.yaml':
#         file_names.append(dir_id)
#         path = './mlruns/0/'+f'{dir_id}'+'/metrics/r2'
#         fp = open(path)
#         data = fp.read()
#         r2_values.append(round(float(data.split(' ')[1]), 5))
        
# # Getting the best model
# idx = np.argmax(r2_values)
# # print(r2_values)
# # print(idx)
# # print(file_names)
# model_dir = os.listdir('./mlruns/0/'+file_names[idx]+'/artifacts/')
# model_path = './mlruns/0/'+file_names[idx]+'/artifacts/'+model_dir[1]+'/model.pkl'
# # print(model_path)
# model = pickle.load(open(model_path, 'rb'))

# Getting the test data samples
X_test, test_data, Y_test = return_test_sample()
X_test = X_test[0:11]
X_test = np.round(X_test)

plt_delta = []
num_runs = 0

@app.route('/')
def home():
    '''
    For rendering home page
    '''
    test_list = []
    for i,k in enumerate(X_test):
        value = list(k)
        test_list.append({f"Sample": i, f"values": value})

    return render_template('index.html', test_list = test_list, best_model='Model is taken from S3 bucket')

@app.route('/predict/<idx>')
def predict(idx):
    '''
    For rendering results on HTML GUI
    '''
    # Getting predictions
    prediction = model.predict(test_data[int(idx)].reshape(1, -1))

    # Measuring the errors and monitoring the model
    delta = (abs(Y_test[int(idx)]-prediction)/abs(Y_test[int(idx)]))

    # Rounding off
    output = prediction[0]*1000
    output = round(output, 3)

    img = io.BytesIO()

    # incrementing the number of runs
    global num_runs
    num_runs += 1

    # Plotting
    plt_delta.append(delta)
    num_runs_list = list(range(1,num_runs + 1))
    plt.plot(num_runs_list, plt_delta, marker='o')
    plt.ylabel('Errors : ΔY')
    plt.xlabel('Run Number')
    plt.title('Model monitoring')

    # Saving the new image
    plt.savefig(img, format='png')
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Displaying the other samples
    test_list = []
    for i,k in enumerate(X_test):
        value = list(k)
        test_list.append({f"Sample": i, f"values": value})

    return render_template('results.html', 
        test_data = X_test[int(idx)], 
        test_list = test_list, 
        delta = round(delta[0], 3), 
        plot_img = f'data:image/png;base64,{plot_url}', 
        prediction_text=f'Predicted House price : $ {output}', 
        best_model='Model is taken from S3 bucket')


@app.route('/retrain', methods=['POST', 'GET'])
def retrain():
    '''
    For retraining the model
    '''
    r2 = None
    if request.method == 'POST':
    
        inputs = list(request.form.values())
        print(inputs)
        print(type(inputs))
            
        if inputs[0] == 'Linear Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, alpha=None, 
                            l1_ratio=None, learning_rate=None, n_estimators=None, max_depth=None)
            

        elif inputs[0] == 'SGD Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=inputs[1], 
                            alpha=float(inputs[2]), l1_ratio=float(inputs[3]), 
                            learning_rate=inputs[4], n_estimators=None, max_depth=None)

        elif inputs[0] == 'RF Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=None, 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'DT Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=None, 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'GBDT Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=float(inputs[3]), 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

        elif inputs[0] == 'XGB Regression':
            # Calling the function for retraining
            r2 = train_fcn(exp_name='Compare_algo', model_name=inputs[0], penalty=None, 
                            alpha=None, l1_ratio=None, learning_rate=float(inputs[3]), 
                            n_estimators=int(inputs[1]), max_depth=int(inputs[2]))

    # text=f'The R_sqaure value after training is {r2}'

    return render_template('retrain.html', text=f'The R_sqaure value after training is {r2}', best_model='Model is taken from Local')


# Test API
@app.route('/retrain_test', methods=['POST'])
def retrain_test():

    # Getting requests
    content = request.get_json(force=True)
    query_data = content['query_data']

    print(query_data)
    print(content)

    outputs = query_data

    return outputs


# if __name__ == "__main__":

#     # Debug/Development
#     logger.debug("Starting Flask Server")
#     app.run(host='0.0.0.0', port='8200' ,debug=True)

    # Production
    # Keep WSGIServer(('', 8200), app) while prduction
#    logger.debug("Server running at http://127.0.0.1:8200/")
#    http_server = WSGIServer(('', 8200), app)
#    http_server.serve_forever()
