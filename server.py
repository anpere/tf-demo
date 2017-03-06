from flask import Flask, request, jsonify
import sys, socket
import tensorflow as tf
'''
 setup a server that handles requests made by clients
  - listens for training points
  - will also classify points
'''

app = Flask(__name__)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

@app.route("/test", methods=["GET"])
def test():
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print (sess.run(accuracy, feed_dict={x: mnist.test.images,
                                         y_: mnist.test.labels}))
    return jsonify({"ok": "ok", "accuracy":"TODO"})
@app.route("/classify", methods=["GET"])
def classify():
    vector = request.json["vector"]
    return jsonify({"ok":"ok"})
@app.route("/train", methods=["POST"])
def train():
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
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
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
        
