# A simple implementation of BackPropagation Neural Network in tensorflow framework

import tensorflow as tf
import numpy as np


def create_nn_layer(inputs, input_size, output_size, activation_function=None):
    layer_weights = tf.Variable(tf.random_normal([input_size, output_size]))
    layer_bias = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    linear_trans_result = tf.matmul(inputs, layer_weights) + layer_bias
    if activation_function is None:
        return linear_trans_result
    else:
        return activation_function(linear_trans_result)


class BPNeuralNetwork:
    def __init__(self, session):
        self.session = session

        self.loss = None
        self.optimizer = None

        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.num_hidden_layers = 0
        self.labels = []

        self.input_dim = 0
        self.output_dim = 0
        self.hidden_dims = []

    def setup(self, each_layer_dims):
        if (len(each_layer_dims) < 3):
            return

        # set each layer's dimension
        self.input_dim = each_layer_dims[0]
        self.num_hidden_layers = len(each_layer_dims) - 2
        self.hidden_dims = each_layer_dims[1:-1]
        self.output_dim = each_layer_dims[-1]

        # build the input and output layer
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.output_dim])

        # build those hidden layers
        temp_input_size = self.input_dim
        temp_output_size = self.hidden_dims[0]
        self.hidden_layers.append(create_nn_layer(self.input_layer, temp_input_size, temp_output_size,
                                                  activation_function=tf.nn.relu))
        for i in range(self.num_hidden_layers - 1):
            temp_input_size = temp_output_size
            temp_output_size = self.hidden_dims[i + 1]
            temp_inputs = self.hidden_layers[i]
            self.hidden_layers.append(create_nn_layer(temp_inputs, temp_input_size, temp_output_size,
                                                      activation_function=tf.nn.relu))

        # build the output layer
        self.output_layer = create_nn_layer(self.hidden_layers[-1], self.hidden_dims[-1], self.output_dim,
                                            activation_function=None)

    def train(self, instances, labels, max_training_iterations=100, learning_rate=1e-2):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(np.array(self.labels) - np.array(self.output_layer)),
                                                 reduction_indices=[0]))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

        for i in range(max_training_iterations):
            _, loss_value = self.session.run([self.optimizer, self.loss], feed_dict={self.input_layer: instances, self.labels: labels})
            # print each iteration's loss
            print(i, loss_value)

        # save the trained model in a directory named "model"
        saver = tf.train.Saver()
        saver.save(self.session, "./model/model.ckpt")

    # inference
    def forward(self, instance):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: instance})

    # test the exclusive or (XOR) logic
    def test(self):
        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_data = np.array([[0, 1, 1, 0]]).transpose()
        test_data = np.array([[0, 1]])
        self.setup([2, 10, 5, 1])
        self.train(x_data, y_data)
        print(self.forward(test_data))


if __name__ == '__main__':
    with tf.Session() as sess:
        model = BPNeuralNetwork(sess)
        model.test()