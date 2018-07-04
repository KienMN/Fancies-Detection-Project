from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from lvq_network import LvqNetworkWithNeighborhood
import pymongo
from pymongo import MongoClient
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
import json

# Initializing connection with mongodb database
client = MongoClient('localhost', 27017)
database = client.neural_network
collection = database.lvq

app = Flask(__name__)
api = Api(app)

class LVQTrainAPI(Resource):
  def __init__(self):
    self.reqparse = reqparse.RequestParser()
    self.reqparse.add_argument('model_id', type = str, required = True, location = 'json')
    self.reqparse.add_argument('n_rows', type = int, required = True, location = 'json')
    self.reqparse.add_argument('n_cols', type = int, required = True, location = 'json')
    self.reqparse.add_argument('learning_rate', type = float, required = True, location = 'json')
    self.reqparse.add_argument('decay_rate', type = float, required = True, location = 'json')
    self.reqparse.add_argument('neighborhood', type = str, required = True, location = 'json')
    self.reqparse.add_argument('sigma', type = float, default = 1, location = 'json')
    self.reqparse.add_argument('sigma_decay_rate', type = float, default = 1, location = 'json')
    self.reqparse.add_argument('bias', type = bool, default = True, location = 'json')
    self.reqparse.add_argument('weights_initialization', type = str, default = "random", location = 'json')
    self.reqparse.add_argument('X', required = True, location = 'json', action = "append")
    self.reqparse.add_argument('y', required = True, location = 'json', action = "append")
    self.reqparse.add_argument('num_iteration', type = int, required = True, location = 'json')
    self.reqparse.add_argument('epoch_size', type = int, required = True, location = 'json')

  def post(self):
    # Request's arguments
    args = self.reqparse.parse_args()
    
    # Training dataset
    X = args['X']
    y = args['y']
    X = [ast.literal_eval(x) for x in X]
    y = [ast.literal_eval(i) for i in y]
    X = np.array(X)
    y = np.array(y)

    # Number of class
    n_class = len(np.unique(y))

    # Number of features in the dataset
    n_feature = X.shape[1]
    
    # Model id
    model_id = args['model_id']

    lvq = LvqNetworkWithNeighborhood(n_feature = n_feature, n_rows = args['n_rows'], n_cols = args['n_cols'], n_class = n_class,
                                    learning_rate = args['learning_rate'], decay_rate = args['decay_rate'],
                                    sigma = args['sigma'], sigma_decay_rate = args['sigma_decay_rate'],
                                    neighborhood = args['neighborhood'])

    lvq.train_batch(X, y, num_iteration = args['num_iteration'], epoch_size = args['epoch_size'])

    lvq_properties = {
      "n_rows": args.get('n_rows'),
      "n_cols": args.get('n_cols'),
      "competitive_layer_weights": lvq._competitive_layer_weights.tolist(),
      "linear_layer_weights": lvq._linear_layer_weights.tolist()
    }

    collection.update_one({"model_id": model_id}, {"$set": lvq_properties}, upsert = True)
    return {"status": "OK"}, 200
    

class LVQPredictAPI(Resource):
  def __init__(self):
    self.reqparse = reqparse.RequestParser()
    self.reqparse.add_argument('model_id', type = int, required = True, location = 'json')
    self.reqparse.add_argument('X', required = True, location = 'json', action = 'append')

  def post(self):
    args = self.reqparse.parse_args()
    print(args.get("model_id"))
    properties = collection.find_one({"model_id": str(args.get("model_id"))})
    competitive_layer_weights = np.array(properties.get('competitive_layer_weights'))
    linear_layer_weights = np.array(properties.get('linear_layer_weights'))
    n_rows = properties.get("n_rows")
    n_cols = properties.get("n_cols")
    n_feature = competitive_layer_weights.shape[1]
    n_class = linear_layer_weights.shape[0]
    lvq = LvqNetworkWithNeighborhood(n_rows = n_rows, n_cols = n_cols, n_feature = n_feature, n_class = n_class)
    lvq._competitive_layer_weights = competitive_layer_weights
    lvq._linear_layer_weights = linear_layer_weights
    X = args['X']
    X = [ast.literal_eval(x) for x in X]
    X = np.array(X)
    # print(X)
    y_pred = lvq.predict(X)
    response = {}
    response['y_pred'] = y_pred.tolist()
    response = json.dumps(response)
    return response, 200

api.add_resource(LVQTrainAPI, '/api/v1.0/lvq/train')
api.add_resource(LVQPredictAPI, '/api/v1.0/lvq/predict')

if __name__ == "__main__":
  app.run(port=1234)