import tensorflow as tf
import numpy as np
def dnn(input_x, input_y, test_x, test_y):
    # Parameters
    learning_rate = 0.001
    training_epochs = 15000000
    display_step = 1
    batch_size = 1000

    # Network Parameters
    n_hidden_1 = 2560 # 1st layer number of neurons
    n_hidden_2 = 2560 # 2nd layer number of neurons
    n_hidden_3 = 1280
    n_input = input_x.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = input_y.shape[1] # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def biases_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W, ksize):
        return tf.nn.conv2d(x, W, strides=[1, 1, ksize, 1], padding='SAME')

    def max_pool_1x2(x):
        return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    x_image = tf.reshape(X, [-1, 1, n_input, 1])

    W_conv1 = weight_variable([1, 2, 1, 32])
    b_conv1 = biases_variable([32])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1, 30) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)

    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = biases_variable([64])
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2,2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    W_fc1 = weight_variable([9*64, 100])
    b_fc1 = biases_variable([100])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 64*9])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1)

    W_fc2 = weight_variable([100, n_classes])
    b_fc2 = biases_variable([n_classes])
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Construct model

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(logits, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        saver.restore(sess, "./DNN_Model/DNN-22030")

        # Training cycle
        batch = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: input_x[batch:batch+batch_size],
                                                            Y: input_y[batch:batch+batch_size]})
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
                print("Accuracy:", accuracy.eval({X: test_x[:500], Y: test_y[:500]}))
            batch = (batch+batch_size)%input_x.shape[0]

        print("Optimization Finished!")

        # Test model


x_list = np.load('x_list.npy')
y_list = np.load('y_list.npy')

dnn(input_x = x_list[:20000],input_y = y_list[:20000],test_x=x_list[20000:],test_y=y_list[20000:])