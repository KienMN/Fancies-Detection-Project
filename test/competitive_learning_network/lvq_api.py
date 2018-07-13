from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from lvq_network import LvqNetworkWithNeighborhood
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os

app = Flask(__name__)
api = Api(app)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response

class LVQTrainAPI(Resource):

  # def options (self):
  #   return {'Allow' : 'POST' }, 200, \
  #   { 'Access-Control-Allow-Origin': '*', \
  #     'Access-Control-Allow-Methods' : 'POST,GET',
  #     'Access-Control-Allow-Headers': 'Content-Type' }

  def post(self):
    
    
    # Request's data
    json_data = request.get_json()

    # Required parameters
    required_params = ['n_rows', 'n_cols', 'learning_rate', 'decay_rate', 'weights_initialization', 'num_iteration', 'epoch_size']
    optional_neighborhood = ['neighborhood', 'sigma', 'sigma_decay_rate']

    # Model id
    model_id = None
    if json_data.get('model_id'):
      model_id = json_data.get('model_id')
    else:
      return {"message": "No model id is provided"}, 400

    # Parameters
    params = {}
    if not json_data.get('params'):
      return {"message": "No params is provided"}, 400
    # Required parameters
    for param in required_params:
      if json_data.get('params').get(param):
        params[param] = json_data.get('params').get(param)
      else:
        return {"message": "No %s is provided"%param}, 400
    # Optional parameters
    if json_data.get('params').get('neighborhood'):
      for param in optional_neighborhood:
        if json_data.get('params').get(param):
          params[param] = json_data.get('params').get(param)
        else:
          return {"message": "No %s is provided"%param}, 400
    else:
      params['neighborhood'] = None
      params['sigma'] = 0
      params['sigma_decay_rate'] = 1

    # Training dataset
    if not json_data.get('train'):
      return {"message": "No dataset is provided"}, 400
    X_train = None
    y_train = None
    if json_data.get('train').get('data') and json_data.get('train').get('target'):
      X_train = json_data.get('train').get('data')
      y_train = json_data.get('train').get('target')
    else:
      return {"message": "Dataset is lack of training set or target set"}, 400
    X_train = np.array(X_train)
    X_train = X_train.T
    y_train = np.array(y_train).astype(np.int8)
    
    # Data preprocessing
    sc = None
    from sklearn.preprocessing import MinMaxScaler
    if params.get('weights_initialization') == 'pca':  
      sc = MinMaxScaler(feature_range = (-1, 1))
    else:
      sc = MinMaxScaler(feature_range = (0, 1))
    X_train = sc.fit_transform(X_train)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    
    # Number of class
    n_class = len(np.unique(y_train))

    # Number of features in the dataset
    n_feature = X_train.shape[1]
    
    # Training the LVQ model
    try:
      lvq = LvqNetworkWithNeighborhood(n_feature = n_feature, n_rows = params.get('n_rows'), n_cols = params.get('n_cols'),
                                      n_class = n_class,
                                      learning_rate = params.get('learning_rate'), decay_rate = params.get('decay_rate'),
                                      sigma = params.get('sigma'), sigma_decay_rate = params.get('sigma_decay_rate'),
                                      neighborhood = params.get('neighborhood'))

      if params.get('weights_initialization') == 'pca':
        lvq.pca_weights_init(X_train)
      elif params.get('weights_initialization') == 'sample':
        lvq.sample_weights_init(X_train)

      lvq.train_batch(X_train, y_train, num_iteration = params.get('num_iteration'), epoch_size = params.get('epoch_size'))
    except TypeError:
      return {"message": "Some data in the body has wrong type"}, 400

    # Dumping the models
    from sklearn.externals import joblib
    lvq_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + ".sav"
    scaler_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + "_scaler.sav"
    label_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + "_label.sav"
    joblib.dump(lvq, lvq_model_filepath)
    joblib.dump(sc, scaler_model_filepath)
    joblib.dump(label_encoder, label_model_filepath)
    
    return {'message': 'success', 'status': 200}, 200
    

class LVQPredictAPI(Resource):
  def __init__(self):
    self.reqparse = reqparse.RequestParser()
    self.reqparse.add_argument('model_id', type = int, required = True, location = 'json')
    self.reqparse.add_argument('X', required = True, location = 'json', action = 'append')

  def post(self):
    # Request's data
    json_data = request.get_json()
    
    # Model_id
    model_id = None
    if json_data.get('model_id'):
      model_id = json_data.get('model_id')
    else:
      return {"message": "No model id is provided"}, 400

    # Predicting dataset
    X_pred = None
    if json_data.get('data'):
      X_pred = json_data.get('data')
    else:  
      return {"message": "No dataset is provided"}, 400
    X_pred = np.array(X_pred)
    X_pred = X_pred.T
    
    # Loading trained model
    from sklearn.externals import joblib
    lvq = None
    sc = None
    label_encoder = None
    try:
      lvq_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + ".sav"
      scaler_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + "_scaler.sav"
      label_model_filepath = os.path.dirname(os.getcwd()) + '/dump_model/' + model_id + "_label.sav"
      lvq = joblib.load(lvq_model_filepath)
      sc = joblib.load(scaler_model_filepath)
      label_encoder = joblib.load(label_model_filepath)
    except:
      return {"message": "Can not found model id"}, 400
    X_pred = sc.transform(X_pred)
    y_pred = lvq.predict(X_pred).astype(np.int8)
    y_pred = label_encoder.inverse_transform(y_pred)
    
    # Generating response
    response = {}
    response['target'] = y_pred.tolist()
    response['message'] = 'success'
    response = json.dumps(response)
    response['status'] = 200
    # print(response)
    return response, 200

api.add_resource(LVQTrainAPI, '/api/v1.0/lvq/train')
api.add_resource(LVQPredictAPI, '/api/v1.0/lvq/predict')

if __name__ == "__main__":
  app.run(host = '0.0.0.0', port = 1234)