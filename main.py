from flask import Flask
import numpy as np
import tensorflow as tf
from flask import make_response
import os
import json
from flask import Flask, jsonify
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


def nice_json(arg):
    response = make_response(json.dumps(arg, sort_keys=True, indent=4))
    response.headers['Content-type'] = "application/json"
    return response


@app.route("/")
def hello():
    return "Hello World from Flask"


@app.route('/sumtwo/<num1>/<num2>')
def sumnum(num1, num2):
    """
    sumtwonumber API
    This API return sum of two numbers
    ---
    tags:
      - MLApi
    parameters:
      - name: num1
        in: path
        type: int
        required: true
      - name: num2
        in: path
        type: int
        required: true
    responses:
      200:
        description: A single user item
        schema:
          id: user_response
          properties:
            sumresult:
              type: string
              description:  result 
              default: some_result
    """
    num3 = int(num1) + int(num2)
    # return "sum of two  number is  : " + str(num3)
    return jsonify({'sumresult': str(num3)})


@app.route('/dotvec/<vector01>/<vector02>')
def dotproduct(vector01, vector02):
    """
    vectordotproduct API
    This API returns dot product of two vector
    ---
    tags:
      - MLApi
    parameters:
      - name: vector01
        in: path
        type: string
        description: input vector
        required: true
      - name: vector02
        in: path
        type: string
        description: input vector
        required: true
    responses:
      200:
        description: A single user item
        schema:
          id: user_response
          properties:
            result_vector:
              type: string
              description: dot product result
              default: "[...]"
            vector_01:
              type: string
              description: input vector
              default: "[...]"
            vector_02:
              type: string
              description: input vector
              default: "[...]"
    """
    snum1 = str(vector01)
    snum2 = str(vector02)
    np_num1 = np.array(snum1[1:len(snum1) - 1].split(","), dtype='|S4')
    np_num1 = np_num1.astype(np.int)
    np_num2 = np.array(snum2[1:len(snum2) - 1].split(","), dtype='|S4')
    np_num2 = np_num2.astype(np.int)
    print np_num1, np_num1.size, np_num1.dtype
    print np_num2, np_num2.size, np_num2.dtype
    print np_num1 * np_num2
    with tf.Session() as ses:
        var1 = tf.Variable(np_num1)
        var2 = tf.Variable(np_num2)
        var3 = var1 * var2
        tf.initialize_all_variables().run()
        var4 = ses.run(var3)
    jsonvar = "{'vector_01':" + snum1 + " ,'vector_02':" + snum2 + " ,'result_vector':" + str(var4.tolist()) + "}"
    return nice_json(jsonvar)
    # return "Tensorflow dot product of tow vector is  : " + str(0) +":"+ str(var4)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80)

