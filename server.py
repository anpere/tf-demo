from flask import Flask, request, jsonify
import sys, socket
import tensorflow as tf
import logging

'''
 setup a server that handles requests made by clients
  - listens for training points
  - will also classify points
'''

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
counter = 0
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

@app.route("/test", methods=["GET"])
def test():
    ## TODO: not really sure who should access test data...
    print("Testing")
    test_images = request.json["test_image"]
    test_labels = request.json["test_labels"]
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_images,
                                         y_: test_labels}))
    print("DONE")
    return jsonify({"ok": "ok", "accuracy":"TODO"})
@app.route("/classify", methods=["GET"])
def classify():
    vector = request.json["vector"]
    return jsonify({"ok":"ok"})
@app.route("/train", methods=["POST"])
def train():
    global counter
    counter +=1
    vector = request.json["training_point"]
    label = request.json["classification"]
    sess.run(train_step, feed_dict={x: vector, y_: label})
    return jsonify({"ok":"ok"})

if __name__ == "__main__":
    """Usage: server.py <host> <port>"""
    ##  does the numerical heavy lifting
    ## which requires model setup
    ## follow tensorflow mnist tutorial for background
    ## of the code written here
    ## tensorflow.org/getting_started/mnist/beginners
    ## set up the model

    # define loss and optimizer

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    try:
        app.run(host="0.0.0.0",debug=True)
    except socket.error as err:
        if err.errno == errno.EADDRINUSE:
            print("Error, could not start server on port: ", port)
            sys.exit(-1)
        else:
            raise err
        
