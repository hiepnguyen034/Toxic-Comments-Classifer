from flask import Flask,url_for,request
import urllib.parse
import timeit
import json
import requests
import sys
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pickle

#sys.path.append(os.path.abspath("./Users/Hiep Nguyen/"))


app = Flask(__name__)

def check_text(text):
    text=[text]
    with open('tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    text = tok.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=150)
    model=load_model('toxic_classifier_model.h5')
    result=model.predict(text)
    if result[0][0]>=0.88 or result[0][1]>=0.7 :
        status='The message might be inappropriate.'
    elif result[0][4]==0.7 or result[0][5]>=0.7:
        status='The message includes offensive contents'
    else:
        status='This message is clean'
    return status

def return_json(text,status):
    dict={'text':text,
          'response':status}
    json_var=json.dumps(dict)
    return json_var

@app.route('/scan/text/', methods=["GET"])
def content_handler():
    text = request.args.get('text')
    status=check_text(text)
    result_dict={'text':text,
          'response':status}
    result_json = json.dumps(result_dict)

    return result_json

if __name__ == '__main__':
    app.run(debug=True)
