import requests
import argparse
import simplejson
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
''' send training points to the server
    will batch training points and send them to a server
    will then ask the server to classify some points
    and everything will be awesome!
'''

def send_training_point(image_vector, label):
    payload = {"training_point":simplejson.dumps(image_vector.tolist()),
               "classification":simplejson.dumps(label.tolist())
    }

    res = requests.post(url("train"), json=payload)

def test_server():
    ## fetch the test data from the mnist data-set, and send to
    ## the server with the intent of measuring the accuracy of the model
    payload = {"test_image": simplejson.dumps(mnist.test.images.tolist()),
               "test_labels": simplejson.dumps(mnist.test.labels.tolist())
               }
    res = requests.get(url("test"), json=payload)
    print(res)
    resj = res.json()
    print(resj)
    print(res["accuracy"])

def make_training_point(seed):
    ## returns image, and returns label
    return mnist.train.next_batch(100)

def url(endpoint):
    return "http://0.0.0.0:5000/{}".format(endpoint)


if __name__ == '__main__':
    ## More or less follow the mnist_softmax tutorial...
    ## the difference being that the server does all the training and such
    ## while the client provides the server with all the data

    ## at this point the tutorial created the model, but we leave that to the server
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print("training server")
    ## fetch training points from the mnist dataset, and send them
    ## to the server
    for i in range(10):
        image_vector, label = make_training_point(i)
        send_training_point(image_vector, label)
    print("testing server")
    test_server()

