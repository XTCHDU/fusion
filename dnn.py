import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from functools import reduce


def dnn(input_x, input_y, test_x, test_y):
    # Parameters
    learning_rate = 0.000001
    training_epochs = 15000000
    display_step = 1
    batch_size = 500


    n_input = input_x.shape[1]  # MNIST data input (img shape: 28*28)
    n_classes = input_y.shape[1]  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input], name="X")
    Y = tf.placeholder("float", [None, n_classes], name="Y")

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

    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    Y = tf.placeholder(tf.float32, [None, n_classes])
    x_image = tf.reshape(X, [-1, 1, n_input, 1])

    W_conv1 = weight_variable([1, 2, 1, 32])
    b_conv1 = biases_variable([32])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1, 30) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)

    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = biases_variable([64])
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    W_conv3 = weight_variable([1, 3, 64, 128])
    b_conv3 = biases_variable([128])
    h_conv3 = tf.nn.relu(conv1d(h_pool2, W_conv3, 1) + b_conv3)

    W_conv4 = weight_variable([1, 3, 128, 256])
    b_conv4 = biases_variable([256])
    h_conv4 = tf.nn.relu(conv1d(h_conv3, W_conv4, 1) + b_conv4)

    W_fc1 = weight_variable([9 * 256, 300])
    b_fc1 = biases_variable([300])
    h_pool2_flat = tf.reshape(h_conv4, [-1, 256 * 9])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([300, n_classes])
    b_fc2 = biases_variable([n_classes])
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    tf.add_to_collection('pred_network', logits)

    # Construct model

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(logits, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:

        saver.restore(sess, "./norm_Model/DNN-156000")
        #sess.run(init)
        # Training cycle
        batch = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: input_x[batch:batch + batch_size],
                                                            Y: input_y[batch:batch + batch_size]})
            # Compute average loss
            avg_cost += c
            # Display logs per epoch step
            #if epoch % display_step == 0:
                #print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
            if epoch % 1000 == 0:

                saver.save(sess, "./norm_Model/DNN", global_step=epoch)

                pred = tf.get_collection("pred_network")[0]  # Apply softmax to logits
                # Calculate accuracy
                accuracy = tf.losses.mean_squared_error(Y, pred)
                # print("Accuracy:", logits.eval({X: test_x[:10], Y: test_y[:10]}))
                est = pred.eval({X: test_x, Y: test_y})
                index = 0
                ans = map(mean_squared_error, est, test_y)
                ans_index = 0
                ans_value = 0
                for index, value in enumerate(ans):
                    if value > ans_value:
                        ans_value = value
                        ans_index = index
                print("Acc:", accuracy.eval({X: test_x, Y: test_y}), ans_value)
            batch = (batch + batch_size) % input_x.shape[0]

        print("Optimization Finished!")

        # Test model


x_list = np.load('x_list.npy')
y_list = np.load('y_list.npy')

dnn(input_x = x_list[:40000],input_y = y_list[:40000],test_x=x_list[40000:],test_y=y_list[40000:])
with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('./DNN_Model/DNN-300.meta')
    new_saver.restore(sess, './DNN_Model/DNN-300')
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    y = tf.get_collection('pred_network')[0]

    graph = tf.get_default_graph()

    # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。

    # 使用y进行预测
    origin = x_list[:5]
    origin[:,1000:]*=0.5
    ans = sess.run(y, feed_dict={"Placeholder:0": np.reshape(origin, [-1, 2000])})
    print(ans)
    #print(mean_squared_error(ans,y_list)*9/6)
    import matplotlib.pyplot as plt
    import Model
    for i in range(5):
        testmodel = Model.HammersteinWiener(B=ans[i,:3],b=ans[i,3:6],h=ans[i,6:])
        plt.figure()
        plt.plot(np.array(origin[i][1000:]))
        plt.plot(testmodel.run(np.array(origin[i][:1000])))
        print(mean_squared_error(np.array(origin[i][1000:]),testmodel.run(np.array(origin[i][:1000]))))
    plt.show()
