#TensorFlow demo
This repo contains a very simple client-server
setup that uses tensorflow to train the mnist data-set


## setup
This repo runs with python3, so do make sure to have that
installed
Run `pip install -r requirements.txt` to install all dependencies.
If tensorflow gives you trouble, I suggest visiting https://www.tensorflow.org/install.
I personally couldn't get it to work on Windows 10, it works fine on Linux.
I would start off at mnist_softmax.py, and then look at the client-code.

## mmist_softmax.py
This code was copied from https://github.com/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
The client and server code was appropriately scavenged from this file. 
To follow along with this file, the best source is the tutorial
https://www.tensorflow.org/get_started/mnist/beginners

## client.py
The client has access to the mnist training data. There is nothing preventing
the server from having access to this dataset, but in this example we imagine
that the client provides the server with classified data. As of now
the client only trains the server, as opposed to classifying data-points or
or verifying the accurracy of the model. 

## server.py
The server contains the graph of the model, and performs the neccessary changes
to the graph as the client provides data. It's important that the entire graph
lives in the server because tensorflow performs optimizations that depend
on the graph.
