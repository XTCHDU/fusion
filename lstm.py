import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

def lstm(input_x, input_y, test_x, test_y):
    # Parameters
    learning_rate = 0.001
    training_epochs = 15000000
    display_step = 1
    batch_size = 20

    num_input = input_x.shape[1]  # MNIST data input (img shape: 28*28)
    timesteps = 2000  # timesteps
    num_hidden = 256  # hidden layer num of features
    num_classes = input_y.shape[1]  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps ,1])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(logits, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        batch = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x = input_x[batch:batch+batch_size]
            batch_x = batch_x.reshape((batch_size,2000,1))
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: input_y[batch:batch+batch_size]})
            # Compute average loss
            avg_cost += c
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            if epoch % 100 ==0:

                saver.save(sess, "./lstm/lstm", global_step=epoch)
                pred = logits  # Apply softmax to logits
                # Calculate accuracy
                accuracy = tf.losses.mean_squared_error(Y, pred)
                batch_x = test_x[:1000]
                batch_x = batch_x.reshape((1000, 2000, 1))
                print("Accuracy:", accuracy.eval({X: batch_x, Y: test_y[:1000]}))
            batch = (batch+batch_size)%input_x.shape[0]

        print("Optimization Finished!")

        # Test model


x_list = np.load('x_list.npy')
y_list = np.load('y_list.npy')

lstm(input_x = x_list[:20000],input_y = y_list[:20000],test_x=x_list[20000:],test_y=y_list[20000:])