import tensorflow as tf
import numpy as np
def dnn(input_x, input_y, test_x, test_y):
    # Parameters
    learning_rate = 0.01
    training_epochs = 15000000
    display_step = 1

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    n_hidden_3 = 128
    n_input = input_x.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = input_y.shape[1] # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_4 = tf.nn.relu6(layer_3)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.sigmoid(tf.matmul(layer_4, weights['out']) + biases['out'])
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.losses.mean_squared_error(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: input_x,
                                                            Y: input_y})
            # Compute average loss
            avg_cost += c
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            if epoch % 10 ==0:

                saver.save(sess, "./DNN_Model/DNN", global_step=epoch)
                pred = logits  # Apply softmax to logits
                # Calculate accuracy
                accuracy = tf.losses.mean_squared_error(Y, pred)
                print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

        print("Optimization Finished!")

        # Test model


x_list = np.load('x_list.npy')
y_list = np.load('y_list.npy')

dnn(input_x = x_list[:150000],input_y = y_list[:150000],test_x=x_list[150000:],test_y=y_list[150000:])