import numpy as np

# Create an activation function dictionary from which the activation functions are chosen
def activation(x, func_name):
    if func_name == 'relu':
        return np.maximum(0, x)
    elif func_name == 'relu_leaky':
        return np.maximum(0.01 * x, x)
    elif func_name == 'logsig':
        return 1 / (1 + np.exp(-x))
    elif func_name == 'tanh':
        return np.tanh(x)
    elif func_name == 'linear':
        return x

# Create the corresponding activation function's derivative dictionary
def activation_derivative(x, func_name):
    if func_name == 'relu':
        return np.where(x >= 0, 1, 0)
    elif func_name == 'relu_leaky':
        return np.where(x >= 0, 1, 0.01)
    elif func_name == 'logsig':
        logs = 1 / (1 + np.exp(-x))
        return logs * (1 - logs)
    elif func_name == 'tanh':
        return 1 - np.square(np.tanh(x))
    elif func_name == 'linear':
        return np.ones([len(x),1])                                                                                      # Return an array of 1 instead of a single 1

# Initialize weight, first and second moment matrices for all layers
def parameter_generate(layer, train):
    weight, mAd, vAd = {}, {}, {}                                                                                       # Create dictionaries to store information of different layers
    input_dim = np.shape(train)[0]                                                                                      # Dimension of training input
    for i in range(len(layer)):
        if i == 0:                                                                                                      # Weight matrix for the input layer
            weight[i] = np.random.uniform(-1, 1, [layer[i], input_dim])
        else:
            weight[i] = np.random.uniform(-1, 1, [layer[i], layer[i - 1] + 1])                                          # Weight matrices for the hidden and the output layers
    for i in range(len(weight)):
        mAd[i] = vAd[i] = np.zeros(weight[i].shape)                                                                     # Initialize first and second order moments to zero according to the size of the weight matrices
    return weight, mAd, vAd

# Forward propagation, generate a prediction based on a set of current weight
def forward_propagation(weight, train, act_func, column):
    v, xo = {}, {}                                                                                                      # Store Net activation and activation in dictionaries for Back Propagation
    input_dim = np.shape(train)[0]
    for i in range(len(weight)):
        if i == 0:                                                                                                      # Net activation for the first hidden layer
            v[i] = weight[i] @ np.reshape(train[:, column], [input_dim, 1])                                             # Reshape explicitly to maintain the vector property of the training input
        else:                                                                                                           # Net activation for rest of the layers
            v[i] = weight[i] @ np.reshape(xo[i - 1],[len(xo[i - 1]), 1])                                                #'@' represent matrix multiplication
        if i != len(weight) - 1:                                                                                        # Activation for the hidden layers
            xo[i] = np.insert(activation(v[i], act_func[i]),0,1)                                                        # Include the bias '1' in every activation
        else:                                                                                                           # Activation for the output layer
            xo[i] = activation(v[i], act_func[i])                                                                       # Exclude the bias '1' for the output layer
    return v, xo

# Stochastic Gradient Descent
def SGD(weight, layer, train, traintarget, v, xo, column, eta, act_func):
    input_dim = np.shape(train)[0]
    for i in range(len(layer) - 1, -1, -1):
        if i == len(layer) - 1:                                                                                                                                             # Local gradient for the output layer neurons
            delta = (np.reshape(traintarget[:, column], [len(traintarget[:, column]), 1]) - xo[i]) * np.reshape(activation_derivative(v[i], act_func[i]), [len(v[i]), 1])
        else:                                                                                                                                                               # Local gradient for the hidden layer neurons: current layer local gradient = last layer weight(excluding bias) transposed @ last layer local gradient x current layer activation functions' derivative
            delta = (weight[i + 1][:, 1:].T @ delta) * np.reshape(activation_derivative(v[i], act_func[i]), [len(v[i]), 1])
        if i != 0:                                                                                                                                                          # Update the weight in the output and the hidden layers
            weight[i] = weight[i] + eta * delta @ np.reshape(xo[i - 1], [1, len(xo[i - 1])])                                                                                # Equivalent to the cross product of the current layer local gradient and the previous layer activation
        else:                                                                                                                                                               # Update the weight for the input layer
            weight[i] = weight[i] + eta * delta @ np.reshape(train[:, column],[1, input_dim])
    return weight

# Adaptive Moment Estimation
def Adam(weight, mAD, vAD, layer, train, traintarget, v, xo, column, eta, beta1Adam, beta2Adam, epsAdam, act_func, epoch):
    input_dim = np.shape(train)[0]
    for k in range(len(layer) - 1, -1, -1):
        if k == len(layer) - 1:                                                                                                                                             # Local gradient for the output layer neurons
            delta = (np.reshape(traintarget[:, column], [len(traintarget[:, column]), 1]) - xo[k]) * activation_derivative(v[k], act_func[k])
        else:                                                                                                                                                               # Local gradient for the hidden layer neurons: current layer local gradient = last layer weight(excluding bias) transposed @  layer local gradient x current layer activation functions' derivative
            delta = (weight[k + 1][:, 1:].T @ delta) * np.reshape(activation_derivative(v[k], act_func[k]), [len(v[k]), 1])
        if k != 0:                                                                                                                                                          # Update first and second order moments for the output and the hidden layers
            mAD[k] = beta1Adam * mAD[k] + (1 - beta1Adam) * - delta @ np.reshape(xo[k - 1], [1, len(xo[k - 1])])                                                            # m = beta1 * m + (1-beta1) * gradient
            vAD[k] = beta2Adam * vAD[k] + (1 - beta2Adam) * np.square(- delta @ np.reshape(xo[k - 1], [1, len(xo[k - 1])]))                                                 # v = beta2 * v + (1-beta2) * gradient^2
        else:                                                                                                                                                               # Update first and second order moments for  the input layers
            mAD[k] = beta1Adam * mAD[k] + (1 - beta1Adam) * - delta @ np.reshape(train[:, column], [1, input_dim])
            vAD[k] = beta2Adam * vAD[k] + (1 - beta2Adam) * np.square(- delta @ np.reshape(train[:, column], [1, input_dim]))
        mcorr = mAD[k] / (1 - beta1Adam ** epoch)                                                                                                                           # De-biasing terms
        vcorr = vAD[k] / (1 - beta2Adam ** epoch)
        weight[k] = weight[k] - (eta / np.sqrt(vcorr + epsAdam)) * mcorr                                                                                                    # w = w - eta * mcorr/sqrt(vcorr+e)
    return weight, mAD, vAD

# Model predictions on training and test set, no net activation and activation dictionaries created
def prediction(weight, set, settarget, act_func):
    input_dim = np.shape(set)[0]
    output_dim = np.shape(settarget)[0]                                                                                 # Dimension of the output
    y = np.empty((output_dim, 1))                                                                                       # Initialize model predictions with 2D empty array, the bias is excluded
    for i in range(np.shape(set)[1]):
        for j in range(len(weight)):
            if j == 0:                                                                                                  # Net activation for the first hidden layer
                z = weight[j] @ np.reshape(set[:, i],[input_dim, 1])
            else:                                                                                                       # Net activation for the rest of the layers
                z = weight[j] @ np.reshape(a,[len(a), 1])
            if j != len(weight) - 1:                                                                                    # Activation for the hidden layers
                a = np.insert(activation(z, act_func[j]), 0, 1)
            else:                                                                                                       # Activation for the output layer
                a = activation(z, act_func[j])
        y = np.concatenate((y,a),axis=1)                                                                                # Concatenate all the model predictions (column vectors) horizontally
    y = np.reshape(np.delete(y,0, axis=1),[output_dim, np.shape(set)[1]])                                               # Remove the initial empty column (empty does not mean void, but something in the memory at the time)
    return y

# Train the network with given hyperparameters and optimizers
def train_network(weight, mAD, vAD, layer, train, traintarget, test, testtarget, eta, beta1, beta2, eps, act_func, algorithm, performance, max_epoch):
    costtrain = np.inf                                                                                                  # Cost on training set, set to infinite to enter the while loop
    costtrainrecord, costtestrecord, epochrecord = [], [], []
    e = 1
    n = np.shape(train)[1]                                                                                              # Number of training input
    N = np.shape(test)[1]                                                                                               # Number of test input
    while costtrain > performance and e <= max_epoch:                                                                   # Train the model until the performance is reached
        for i in range(n):
            z, a = forward_propagation(weight, train, act_func, i)                                                      # obtain net activation and activation by forward propagation
            if algorithm == 'SGD':                                                                                      # Select algorithms to train the network and update weights accordingly
                weight = SGD(weight, layer, train, traintarget, z, a, i, eta, act_func)
            elif algorithm == 'Adam':
                weight, mAD, vAD = Adam(weight, mAD, vAD, layer, train, traintarget, z, a, i, eta, beta1, beta2, eps, act_func, e)
        ytr = prediction(weight, train, traintarget, act_func)                                                          # Predictions on training set
        yte = prediction(weight, test, testtarget, act_func)                                                            # Predictions on test set
        costtrain = np.sum(np.square((traintarget - ytr)))
        costtrain = 1 / n * costtrain                                                                                   # Loss (MSE) on the training set
        costtest = np.sum(np.square((testtarget - yte)))
        costtest = 1 / N * costtest                                                                                     # Loss (MSE) on the test set
        costtrainrecord.append(costtrain)
        costtestrecord.append(costtest)
        epochrecord.append(e)
        if e % (max_epoch/10) == 0:                                                                                     # Display the current training progress
            print(e / max_epoch * 100, '%')
        e = e + 1
    print('Epoch number:', e - 1)
    return weight, costtrainrecord, costtestrecord, epochrecord

# Perform regression with the trained network
def regression(train, traintarget, test, testtarget, layer, act_func, eta, beta1, beta2, eps, algorithm, performance, max_epoch):
    weight, mAD, vAD = parameter_generate(layer, train)                                                                                                                                                         # Weight initialisation
    weight, costtrainrecord, costtestrecord, epochrecord = train_network(weight, mAD, vAD, layer, train, traintarget, test, testtarget, eta, beta1, beta2, eps, act_func, algorithm, performance, max_epoch)    # Train the network
    y = prediction(weight, test, testtarget, act_func)                                                                                                                                                          # Prediction on the test set
    return y, costtrainrecord, costtestrecord, epochrecord

# Perform classification with the trained network
def classification(train, trainlabel, test, testlabel, layer, act_func, eta, beta1, beta2, eps, algorithm, performance, max_epoch):                                                           # Perform regression with trained network
    weight, mAD, vAD = parameter_generate(layer, train)                                                                                                                                                         # Weight initialisation
    weight, costtrainrecord, costtestrecord, epochrecord = train_network(weight, mAD, vAD, layer, train, trainlabel, test, testlabel, eta, beta1, beta2, eps, act_func, algorithm, performance, max_epoch)      # Train the network
    y = prediction(weight, test, testlabel, act_func)                                                                                                                                                           # Prediction on the test set
    return y, costtrainrecord, costtestrecord, epochrecord
