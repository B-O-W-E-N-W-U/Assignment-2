import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Neural_Network import classification, parameter_generate, prediction # Import the self-adaptive network trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings


warnings.filterwarnings('ignore')
plt.rcParams['axes.grid'] = True                                                                                        
plt.rcParams['font.size'] = 18                                                                                          
plt.rcParams['xtick.labelsize'] = 20                                                                                  
plt.rcParams['ytick.labelsize'] = 20                                                                                   
plt.rcParams['lines.linewidth'] = 2                                                                                    
np.random.seed(33)                                                                                               

# Data Preprocessing
def iris_data_preprocess(encoding, split_ratio):
    iris = pd.read_csv('IrisData.txt', header=None)
    print(iris.drop_duplicates(iris.columns[-1]))
    print('Iris dataset size:',iris.shape)
    iris = pd.get_dummies(iris, columns=[iris.columns[-1]]) # Encode each iris type with one-hot vector
    rename = {iris.columns[-3]: 'setosa', iris.columns[-2]: 'versicolor', iris.columns[-1]: 'virginica'}
    iris = iris.rename(columns=rename) # Rename the iris to be more concise
    encoder = {True: encoding[0], False: encoding[1]} # Available for user to encode again to avoid 0 and 1
    replace = {'setosa': encoder, 'versicolor': encoder, 'virginica': encoder}
    iris = iris.replace(replace) # Replace 1 with encoding[0] and 0 with encoding[1]
    print('\nEncoded Iris dataset:\n',iris.drop_duplicates(iris.columns[4:-1]))
    iris = iris.sample(frac=1).reset_index(drop=True) # Shuffle the rows of data and delete the old version
    print('\nShuffled Iris dataset:\n',iris.iloc[[0,50,100]])

    irisT = iris.T  # Arrange iris data to column vectors
    num = np.shape(irisT)[1] # Number of the data in the iris dataset
    ratio = round(num * split_ratio) # Set train to test ratio
    iristrain = irisT.iloc[:, :ratio].to_numpy() # Split to train and test sets and convert to numpy arrays
    iristest = irisT.iloc[:, ratio:].to_numpy()
    irisxtrain = iristrain[:4, :] # First 4 rows are the input features
    irisdtrain = iristrain[4:, :] # Rest rows are the encoded labels
    irisxtest = iristest[:4, :]
    irisdtest = iristest[4:, :]

    n = np.shape(irisxtrain)[1] # Number of training input
    N = np.shape(irisxtest)[1] # Number of test input
    irisxtrain = np.vstack((np.ones([1, n]), irisxtrain))  # Stack a row of 1s on top of the train input to represent bias
    irisxtest = np.vstack((np.ones([1, N]), irisxtest)) # Stack a row of 1s on top of the test input to represent bias
    return irisxtrain, irisdtrain, irisxtest, irisdtest

# Plot prediction results 
def classification_plot(weight, train, traintarget, test, testtarget, layer, act_func, eta, beta1,
                        beta2, eps, algorithm, performance, max_epoch):                                       
    costtrainrecordd, costtestrecordd, epochrecordd, cmm = [], [], [], []                                              
    y = prediction(weight, test, testtarget, act_func)
    y_pred = np.argmax(y, axis=0) # Convert predictions to class indices
    y_true = np.argmax(testtarget, axis=0) # Convert test labels to class indices
    cm = confusion_matrix(y_true, y_pred)
    cmm.append(cm)
    for i in range(len(eta)):
        print('Start training:', algorithm[0], eta[i], layer)
        print('training progress:')
        y, costtrainrecord, costtestrecord, epochrecord = classification(train, traintarget, test, testtarget, layer, act_func, eta[i], beta1, beta2, eps, algorithm[0], performance, max_epoch)

        y_pred = np.argmax(y, axis=0)                                                                                   
        y_true = np.argmax(testtarget, axis=0)                                                                         
        cm = confusion_matrix(y_true, y_pred)                                                                      
        costtrainrecordd.append(costtrainrecord), costtestrecordd.append(costtestrecord), epochrecordd.append(epochrecord), cmm.append(cm)
    for i in range(len(eta)):
        plt.plot(epochrecordd[i], costtrainrecordd[i], label=f'lr = {eta[i]}')                                        
        plt.title('Training Set')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig, axes = plt.subplots(1, len(eta)+1, figsize=(15, 5)) # Create new plot for presenting confusion matrices
    if len(cmm) == 1:
        axes = [axes] # Transform axes to a list
    for i, cm in enumerate(cmm): # Plot confusion matrix for every learning rate
        if i == 0:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Seto', 'Vers', 'Virg'])
            disp.plot(cmap=plt.cm.Blues, ax = axes[i], colorbar=False)
            axes[i].set_title('Initial prediction')
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Seto', 'Vers', 'Virg'])
            disp.plot(cmap=plt.cm.Blues, ax = axes[i], colorbar=False)
            axes[i].set_title(f"lr = {eta[i-1]}")
        if i > 0: # Apply y-axis labels for only the first subplot
            axes[i].set_ylabel('') # Clear y-axis label
            axes[i].tick_params(labelleft=False) # Remove y-axis ticks
        axes[i].grid(False)
    fig.suptitle(algorithm[0])
    for i in range(len(eta)):                                                                                          
        print(algorithm[0], eta[i], 'test loss:', costtestrecordd[i][-1])
    plt.show()


def main():
    # === System hyperparameters === #
    algorithm = ['SGD']
    eta = [0.1, 0.01, 0.001, 0.0001]
    max_epoch = 10000
    performance = 1e-4
    layer = [5, 3, 3]
    act_func = ['tanh', 'tanh', 'linear']
    encoding = [1, 0]
    split_ratio = 0.7

    # === Algorithm hyperparameters === #
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # === Regression === #
    irisxtrain, irisdtrain, irisxtest, irisdtest = iris_data_preprocess(encoding, split_ratio)
    weight, mAD, vAD = parameter_generate(layer, irisxtrain)
    classification_plot(weight, irisxtrain, irisdtrain, irisxtest, irisdtest, layer, act_func, eta, beta1, beta2, eps, algorithm, performance, max_epoch)

if __name__ == "__main__":
    main()
