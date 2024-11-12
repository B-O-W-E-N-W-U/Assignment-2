import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore') # Ignore trivial warnings
plt.rcParams['axes.grid'] = True # Turn on the grid for all future plots
np.random.seed(1111) # Use the same random data for future comparison

def weight_generate(low, high, layer, train):
    weight = {}
    for i in range(len(layer)):  # Randomize weights for all layers
        if i == 0:
            weight[i] = np.random.uniform(low, high, [layer[i], np.shape(train)[0] + 1])  # Include bias weight in the overall weight matrix
        else:
            weight[i] = np.random.uniform(low, high, [layer[i], layer[i - 1] + 1])
    return weight


def forward_propagation(weight, train, column_no):
    v, xo = {}, {}
    for i in range(len(weight)):
        if i == 0:
            v[i] = weight[i] @ np.reshape(np.vstack((np.ones(np.shape(train)[1]), train))[:, column_no],[np.shape(train)[0] + 1, 1])  # Input [1;x] for every train data
        else:
            v[i] = weight[i] @ np.reshape(xo[i - 1], [len(xo[i - 1]), 1])  # Shape the output at each layer to match the column of the weight matrix
        if i != len(weight) - 1:
            xo[i] = np.insert(np.tanh(v[i]), 0,1)  # Use tanh for all hidden layers and add the bias '1' in every layer's output
        else:
            xo[i] = 1 * v[i]  # The bias '1' is removed since v for the final layer is already obtained
    return v, xo


def SGD(weight, layer, train, trtarget, v, xo, column_no, eta):
    for k in range(len(layer) - 1, -1, -1):
        if k == len(layer) - 1:
            delta = (np.reshape(trtarget[:, column_no], [len(trtarget[:, column_no]), 1]) - xo[k]) * 1  # The bias input '1' is removed when calculating delta
        else:
            delta = (weight[k + 1][:, 1:].T @ delta) * np.reshape(1 - np.square(np.tanh(v[k])), [len(v[k]), 1])  # So does the bias weights in the first column
        if k != 0:
            weight[k] = weight[k] + (eta * delta * 1) @ np.reshape(xo[k - 1], [1, len(xo[k - 1])])  # Update the weights
        else:
            weight[k] = weight[k] + eta * delta @ np.reshape(np.insert(train[:, column_no], 0, 1), [1, np.shape(train)[0] + 1])  # Don't forget the bias of 1 for the input layer
    return weight


def Adam(weight, momentum, velocity, layer, train, trtarget, v, xo, column_no, eta, epoch):
    beta1 = 0.9 # 'Good' Values to choose according to publications
    beta2 = 0.999
    epsilon = 1e-8
    for k in range(len(layer) - 1, -1, -1):
        if k == len(layer) - 1:
            delta = (np.reshape(trtarget[:, column_no], [len(trtarget[:, column_no]), 1]) - xo[k]) * 1  # The bias input '1' is removed when calculating delta
        else:
            delta = (weight[k + 1][:, 1:].T @ delta) * np.reshape(1 - np.square(np.tanh(v[k])), [len(v[k]), 1])  # So does the bias weights in the first column
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


def train_network(weight, layer, train, trtarget, test, tetarget, eta, performance, algorithm):
    m, ve = {}, {}
    error = np.inf
    e = 0
    for i in range(len(weight)):
        m[i] = np.random.uniform(0, 1, [np.shape(weight[i])[0], np.shape(weight[i])[1]])
        ve[i] = np.random.uniform(0, 1, [np.shape(weight[i])[0], np.shape(weight[i])[1]]) # Ensure that v0 is not negative due to the root on the denominator of adam
    while error > performance and e < 10000: # Train the model until the performance is reached
        for j in range(np.shape(train)[1]):  # Select all the inputs in the same column (in case of multiple inputs)
            v, xo = forward_propagation(weight, train, j)
            if algorithm == 'SGD':
                weight = SGD(weight, layer, train, trtarget, v, xo, j, eta)
            elif algorithm == 'Adam':
                weight = Adam(weight, m, ve, layer, train, trtarget, v, xo, j, eta, e)
        y = prediction(weight,test,tetarget)
        error = 0
        for k in range(np.shape(y)[1]): # Calculate MSE
            for n in range(np.shape(y)[0]):
                error = error + (tetarget[n, k] - y[n, k]) ** 2
        error = 1 / (np.shape(tetarget)[1] * np.shape(tetarget)[0]) * error
        e = e + 1
    print(algorithm,'Mean Square Error:',error)
    print('Epoch number:', e)
    return weight


def regression(layer, performance):
    xtrain = np.array([np.linspace(-1, 1, 41)])
    dtrain = 0.8 * xtrain ** 3 + 0.3 * xtrain ** 2 - 0.4 * xtrain + np.random.normal(0, 0.02, np.shape(xtrain)[1])
    xtest = np.array([np.linspace(-0.97, 0.93, 39)])
    dtest = 0.8 * xtest ** 3 + 0.3 * xtest ** 2 - 0.4 * xtest + np.random.normal(0, 0.02, np.shape(xtest)[1])

    w = weight_generate(-0.5, 0.5, layer, xtrain)
    ws = train_network(w, layer, xtrain, dtrain, xtest, dtest, 0.1, performance, 'SGD')
    ys = prediction(ws, xtest, dtest)
    wa = train_network(w, layer, xtrain, dtrain, xtest, dtest, 0.01, performance, 'Adam')
    ya = prediction(wa, xtest, dtest)

    plt.subplot(1,2,1)
    plt.plot(xtest[0], dtest[0], label='test data', marker='+', color='k', linestyle='None')
    plt.plot(xtest[0], ys[0], label='test prediction')
    plt.xlabel('xtest'), plt.ylabel('Output'), plt.title('SGD')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(xtest[0], dtest[0], label='test data', marker='+', color='k', linestyle='None')
    plt.plot(xtest[0], ya[0], label='test prediction')
    plt.xlabel('xtest'), plt.ylabel('Output'), plt.title('Adam')
    plt.legend()
    plt.show()


def classification(layer, performance):
    iris = pd.read_csv('IrisData.txt', header=None)

    iris = pd.get_dummies(iris, columns=[iris.columns[-1]])  # Same as OneHotEncoder
    rename = {iris.columns[-3]: 'setosa', iris.columns[-2]: 'versicolor',
              iris.columns[-1]: 'virginica'}  # Rename to be more concise
    iris = iris.rename(columns=rename)

    encoder = {True: 0.6, False: -0.6}  # Encode again, for each type of iris with to suit tanh
    replace = {'setosa': encoder, 'versicolor': encoder, 'virginica': encoder}
    iris = iris.replace(replace)

    iris = iris.sample(frac=1).reset_index(drop=True)  # Randomize the rows and delete the old indices

    ratio = round(iris.shape[0] * 0.7)
    iristrain = iris[:ratio]  # Stratify to train and test sets
    iristest = iris[ratio:iris.shape[0]]
    irisxtrain = iristrain.iloc[:, :4].to_numpy().T  # Arrange inputs in columns
    irisdtrain = iristrain.iloc[:, 4:].to_numpy().T
    irisxtest = iristest.iloc[:, :4].to_numpy().T
    irisdtest = iristest.iloc[:, 4:].to_numpy().T

    w = weight_generate(-0.5, 0.5, layer, irisxtrain)
    ws = train_network(w, layer, irisxtrain, irisdtrain, irisxtest, irisdtest, 0.01, performance, 'SGD')
    ys = prediction(ws, irisxtest, irisdtest)
    wa = train_network(w, layer, irisxtrain, irisdtrain, irisxtest, irisdtest, 0.001, performance, 'Adam')
    ya = prediction(wa, irisxtest, irisdtest)
    dsgd, ysgd = [], []
    dadam, yadam = [], []
    for i in range(np.shape(ys)[1]):
        dsgd.append(np.argmax(irisdtest[:, i]))
        ysgd.append(np.argmax(ys[:, i]))
    for i in range(np.shape(ya)[1]):
        dadam.append(np.argmax(irisdtest[:, i]))
        yadam.append(np.argmax(ya[:, i]))
    errors = abs(np.array(dsgd) - np.array(ysgd))
    errora = abs(np.array(dadam) - np.array(yadam))
    accuracys = (1 - np.shape(np.nonzero(errors))[1] / np.shape(ys)[1]) * 100
    accuracya = (1 - np.shape(np.nonzero(errora))[1] / np.shape(ys)[1]) * 100
    print('Accuracy-SGD:', accuracys)
    print('Accuracy-Adam:', accuracya)

    plt.subplot(2, 2, 1)
    plt.plot(dsgd, label='test data', marker='+', color='k', linestyle='None')
    plt.plot(ysgd, 'r', label='test prediction')
    plt.xlabel('xtest index'), plt.ylabel('Iris Encoding'), plt.title('Setosa:0, Versicolor:1, Virginica:2')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(errors, 'bo', label=f'Accuracy:{accuracys:.2f}%')
    plt.xlabel('xtest index'), plt.ylabel('Error'), plt.title('SGD')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(dadam, label='test data', marker='+', color='k', linestyle='None')
    plt.plot(yadam, 'r', label='test prediction')
    plt.xlabel('xtest index'), plt.ylabel('Iris Encoding'), plt.title('Setosa:0, Versicolor:1, Virginica:2')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(errora, 'bo', label=f'Accuracy:{accuracya:.2f}%')
    plt.xlabel('xtest index'), plt.ylabel('Error'), plt.title('Adam')
    plt.legend()
    plt.show()


def main():
    # === Regression === #
    regression([3,1], 5e-4)

    # === Classification === #
    classification([5,3,3],0.035)


if __name__ == "__main__":
    main()
