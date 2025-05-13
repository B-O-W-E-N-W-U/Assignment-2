import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from joblib.backports import access_denied_errors

# warnings.filterwarnings('ignore')  # Ignore trivial warnings
plt.rcParams['axes.grid'] = True  # Turn on the grid for all future plots
np.random.seed(1000-7)  # Use the same random data for future comparison

xtrain = np.array([np.linspace(-1, 1, 41)])
dtrain = 0.8 * xtrain ** 3 + 0.3 * xtrain ** 2 - 0.4 * xtrain + np.random.normal(0, 0.02, np.shape(xtrain)[1])
xtest = np.array([np.linspace(-0.97, 0.93, 39)])
dtest = 0.8 * xtest ** 3 + 0.3 * xtest ** 2 - 0.4 * xtest + np.random.normal(0, 0.02, np.shape(xtest)[1])


def activation(x, func_name): # Generalized activation functions
    if func_name == 'relu':
        return np.maximum(0, x)
    elif func_name == 'leaky_relu':
        return np.maximum(0.01 * x, x)
    elif func_name == 'logsig':
        return 1 / (1 + np.exp(-x))
    elif func_name == 'tanh':
        return np.tanh(x)
    elif func_name == 'linear':
        return x
    else:
        raise ValueError("Unknown activation function: " + func_name)


def activation_derivative(x, func_name): # Generalized derivatives
    if func_name == 'relu':
        return np.where(x >= 0, 1, 0)
    elif func_name == 'leaky_relu':
        return np.where(x >= 0, 1, 0.01)
    elif func_name == 'logsig':
        logs = 1 / (1 + np.exp(-x))
        return logs * (1 - logs)
    elif func_name == 'tanh':
        return 1 - np.square(np.tanh(x))
    elif func_name == 'linear':
        return 1
    else:
        raise ValueError("Unknown activation function: " + func_name)


def weight_generate(low, high, layer, train):
    weight = {}
    for i in range(len(layer)):  # Randomize weights for all layers
        if i == 0:  # Dealing with the input layer
            weight[i] = np.random.uniform(low, high, [layer[i], np.shape(train)[0] + 1])  # Include bias weight in the overall weight matrix
        else:  # For rest of the layers
            weight[i] = np.random.uniform(low, high, [layer[i], layer[i - 1] + 1])
    return weight


def forward_propagation(weight, train, act_func, column_no):
    v, xo = {}, {}
    for i in range(len(weight)):
        if i == 0:  # Dealing with the input layer
            v[i] = weight[i] @ np.reshape(np.insert(train[:, column_no], 0 ,1),[np.shape(train)[0] + 1, 1])  # Add 1 on top of all input and select all the elements in a column
        else:  # For rest of the layers
            v[i] = weight[i] @ np.reshape(xo[i - 1], [len(xo[i - 1]), 1])  # Shape the output at each layer to match the column of the weight matrix
        if i != len(weight) - 1:  # Check if the current layer is the output layer
            xo[i] = np.insert(activation(v[i], act_func[i]), 0,1)  # Use tanh for all hidden layers and add the bias '1' in every layer's output
        else:  # For the output layer
            xo[i] = activation(v[i], act_func[i])  # The bias '1' is removed since v for the final layer is already obtained
    return v, xo


def SGD(weight, layer, train, trtarget, v, xo, column_no, eta, act_func):
    for k in range(len(layer) - 1, -1, -1):
        if k == len(layer) - 1:
            delta = (np.reshape(trtarget[:, column_no], [len(trtarget[:, column_no]), 1]) - xo[k]) * activation_derivative(v[k], act_func[k])  # The bias input '1' is removed when calculating delta
        else:
            delta = (weight[k + 1][:, 1:].T @ delta) * np.reshape(activation_derivative(v[k], act_func[k]), [len(v[k]), 1])  # So does the bias weights in the first column
        if k != 0:
            weight[k] = weight[k] + eta * delta @ np.reshape(xo[k - 1], [1, len(xo[k - 1])])  # Update the weights
        else:
            weight[k] = weight[k] + eta * delta @ np.reshape(np.insert(train[:, column_no], 0, 1), [1, np.shape(train)[0] + 1])  # Don't forget the bias of 1 for the input layer
    return weight


def Adam(weight, momentum, velocity, layer, train, trtarget, v, xo, column_no, eta, act_func, epoch):
    beta1 = 0.9 # 'Good' Values to choose according to publications
    beta2 = 0.999
    epsilon = 1e-8
    for k in range(len(layer) - 1, -1, -1):
        if k == len(layer) - 1:
            delta = (np.reshape(trtarget[:, column_no], [len(trtarget[:, column_no]), 1]) - xo[k]) * activation_derivative(v[k], act_func[k])  # The bias input '1' is removed when calculating delta
        else:
            delta = (weight[k + 1][:, 1:].T @ delta) * np.reshape(activation_derivative(v[k], act_func[k]), [len(v[k]), 1])  # So does the bias weights in the first column
        if k != 0:
            momentum[k] = beta1 * momentum[k] + (1 - beta1) * - delta @ np.reshape(xo[k - 1], [1, len(xo[k - 1])]) # Calculate momentum and velocity
            velocity[k] = beta2 * velocity[k] + (1 - beta2) * np.square(- delta @ np.reshape(xo[k - 1], [1, len(xo[k - 1])]))
        else:
            momentum[k] = beta1 * momentum[k] + (1 - beta1) * - delta @ np.reshape(np.insert(train[:, column_no], 0, 1), [1, np.shape(train)[0] + 1])
            velocity[k] = beta2 * velocity[k] + (1 - beta2) * np.square(delta @ np.reshape(np.insert(train[:, column_no], 0, 1), [1, np.shape(train)[0] + 1]))
        mcorr = momentum[k] / (1 - beta1 ** (epoch + 1)) # Corrected momentum and velocity
        vcorr = velocity[k] / (1 - beta2 ** (epoch + 1))
        weight[k] = weight[k] - (eta / np.sqrt(vcorr + epsilon)) * mcorr
    return weight


def prediction(weight, test, tetarget):
    y = np.zeros((np.shape(tetarget)[0], np.shape(test)[1]))
    for j in range(np.shape(test)[1]):
        for i in range(len(weight)):
            if i == 0:
                v = weight[i] @ np.reshape(np.insert(test[:, j], 0, 1),[np.shape(test[:, j])[0] + 1, 1])  # Input [1;x] for every test data
            else:
                v = weight[i] @ np.reshape(xo, [np.shape(xo)[0], 1])
            if i != len(weight) - 1:
                xo = np.insert(np.tanh(v), 0, 1)
            else:
                xo = 1 * v
        y[:, j] = np.array(xo.T)  # Replace each column of zeros with the predicted output
    return y


def train_network(weight, layer, train, trtarget, test, tetarget, eta, act_func, algorithm, performance):
    m, ve = {}, {}
    error = np.inf
    e = 0
    for i in range(len(weight)):
        m[i] = np.random.uniform(0, 1, [np.shape(weight[i])[0], np.shape(weight[i])[1]])
        ve[i] = np.random.uniform(0, 1, [np.shape(weight[i])[0], np.shape(weight[i])[1]]) # Ensure that v0 is not negative due to the root on the denominator of adam
    while error > performance and e < 10000: # Train the model until the performance is reached
        for j in range(np.shape(train)[1]):  # Select all the inputs in the same column (in case of multiple inputs)
            v, xo = forward_propagation(weight, train, act_func, j)
            if algorithm == 'SGD':
                weight = SGD(weight, layer, train, trtarget, v, xo, j, eta, act_func)
            elif algorithm == 'Adam':
                weight = Adam(weight, m, ve, layer, train, trtarget, v, xo, j, eta, act_func, e)
        y = prediction(weight,test,tetarget)
        error = 0
        for k in range(np.shape(y)[1]): # Calculate MSE
            for n in range(np.shape(y)[0]):
                error = error + np.square((tetarget[n, k] - y[n, k]))
        error = 1 / (np.shape(tetarget)[1] * np.shape(tetarget)[0]) * error
        e = e + 1
    print(algorithm,'Mean Square Error:',error)
    print('Epoch number:', e)
    return weight


def regression(layer, act_func, eta, algorithm, performance):
    weight = weight_generate(-0.5, 0.5, layer, xtrain)
    w = train_network(weight, layer, xtrain, dtrain, xtest, dtest, eta, act_func, algorithm, performance)
    y = prediction(w, xtest, dtest)
    return y


def regression_plot(layer, act_func, eta, algorithm, performance):
    rows = int(np.sqrt(len(algorithm)))
    columns = round(len(algorithm) / rows)
    for i in range(len(algorithm)):
        y = regression(layer, act_func, eta, algorithm[i], performance)
        plt.subplot(rows, columns, i + 1)
        plt.plot(xtest[0], dtest[0], label='test data', marker='+', color='k', linestyle='None')
        plt.plot(xtest[0], y[0], label='test prediction')
        plt.xlabel('xtest'), plt.ylabel('Output'), plt.title(algorithm[i])
        plt.legend()
    plt.show()


def iris_data_preprocess(filename, encoding, split_ratio):
    iris = pd.read_csv(filename, header=None)

    iris = pd.get_dummies(iris, columns=[iris.columns[-1]])  # Same as OneHotEncoder
    rename = {iris.columns[-3]: 'setosa', iris.columns[-2]: 'versicolor',
              iris.columns[-1]: 'virginica'}  # Rename to be more concise
    iris = iris.rename(columns=rename)

    encoder = {True: encoding[0], False: encoding[1]}  # Encode again, for each type of iris with to suit tanh
    replace = {'setosa': encoder, 'versicolor': encoder, 'virginica': encoder}
    iris = iris.replace(replace)

    iris = iris.sample(frac=1).reset_index(drop=True)  # Randomize the rows and delete the old indices

    ratio = round(iris.shape[0] * split_ratio)
    iristrain = iris[:ratio]  # Stratify to train and test sets
    iristest = iris[ratio:iris.shape[0]]
    irisxtrain = iristrain.iloc[:, :4].to_numpy().T  # Arrange inputs in columns
    irisdtrain = iristrain.iloc[:, 4:].to_numpy().T
    irisxtest = iristest.iloc[:, :4].to_numpy().T
    irisdtest = iristest.iloc[:, 4:].to_numpy().T

    return irisxtrain, iristest, irisxtrain, irisdtrain, irisxtest, irisdtest


def classification(layer, act_func, eta, algorithm, encoding, split_ratio, performance):
    irisxtrain, iristest, irisxtrain, irisdtrain, irisxtest, irisdtest = iris_data_preprocess('IrisData.txt', encoding, split_ratio)
    weight = weight_generate(-0.5, 0.5, layer, irisxtrain)
    w = train_network(weight, layer, irisxtrain, irisdtrain, irisxtest, irisdtest, eta, act_func, algorithm, performance)
    yy = prediction(w, irisxtest, irisdtest)

    d, y = [],[]
    for i in range(np.shape(yy)[1]):
        d.append(np.argmax(irisdtest[:, i]))
        y.append(np.argmax(yy[:, i]))
    error = abs(np.array(d) - np.array(y))
    accuracy = (1 - np.shape(np.nonzero(error))[1] / np.shape(yy)[1]) * 100
    return d, y, error, accuracy


def classification_plot(layer, act_func, eta, algorithm, encoding, split_ratio, performance):
    rows = int(np.sqrt(len(algorithm)))
    columns = round(len(algorithm) / rows)
    counter = 1
    for i in range(len(algorithm)):
        d, y, error, accuracy = classification(layer, act_func, eta, algorithm[i], encoding, split_ratio, performance)
        plt.subplot(rows * 2, columns, i + counter)
        counter += 1
        plt.plot(d, label='test data', marker='+', color='k', linestyle='None')
        plt.plot(y, 'r', label='test prediction')
        plt.xlabel('xtest index'), plt.ylabel('Iris Encoding'), plt.title('Setosa:0, Versicolor:1, Virginica:2')
        plt.subplot(rows * 2, columns, i + counter)
        plt.plot(error, 'bo', label=f'Accuracy:{accuracy:.2f}%')
        plt.xlabel('xtest index'), plt.ylabel('Error'), plt.title(algorithm[i])
        plt.legend()
    plt.show()


def main():
    algorithm = ['SGD', 'Adam']
    # === Regression === #
    layer = [3,1]
    act_func = ['tanh','linear']
    eta = 0.01
    performance = 9e-4
    regression_plot(layer, act_func, eta, algorithm, performance)



    # === Classification === #
    layer = [5,3,3]
    act_func = ['tanh','tanh','linear']
    eta = 0.01
    encoding = [0.6, -0.6]
    split_ratio = 0.7
    performance = 0.03
    classification_plot(layer, act_func, eta, algorithm, encoding, split_ratio, performance)


if __name__ == "__main__":
    main()
