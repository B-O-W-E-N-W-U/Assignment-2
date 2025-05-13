import numpy as np
import matplotlib.pyplot as plt
from Neural_Network import regression                                                                                  


plt.rcParams['axes.grid'] = True                                                                                      
plt.rcParams['font.size'] = 18                                                                                          
plt.rcParams['xtick.labelsize'] = 20                                                                     
plt.rcParams['ytick.labelsize'] = 20                                               
plt.rcParams['lines.linewidth'] = 2                                                           
np.random.seed(33)                                                                     

xtrain = np.array([np.arange(-1, 1.05, 0.05)]) # Training input
xtrain = np.array([[x] for x in xtrain.flatten()]) # Reconstruct the array to a column vector where each row represents one input
dtrain = 0.8 * xtrain ** 3 + 0.3 * xtrain ** 2 - 0.4 * xtrain + np.random.normal(0, 0.02, np.shape(xtrain)) # Training target
xtest = np.array([np.arange(-0.97, 1.03, 0.1)]) # Test input
xtest = np.array([[x] for x in xtest.flatten()])
dtest = 0.8 * xtest ** 3 + 0.3 * xtest ** 2 - 0.4 * xtest + np.random.normal(0, 0.02, np.shape(xtest)) # Test target

n = np.shape(xtrain)[0] # Number of training input
N = np.shape(xtest)[0] # Number of test input
xtrainT = xtrain.T # Transpose training input so that each column represent one input
xtrain1 = np.vstack((np.ones([1, n]), xtrainT)) # Stack a row of 1s on top of the training input to represent bias
dtrainT = dtrain.T
xtestT = xtest.T # Transpose test input as well
xtest1 = np.vstack((np.ones([1, N]), xtestT)) # Stack a row of 1s on top of the test input to represent bias
dtestT = dtest.T

# Plot regression results
def regression_plot(train, traintarget, test, testtarget, layer, act_func, eta, beta1, beta2, eps, algorithm, performance, max_epoch): 
    yy, costtrainrecordd, costtestrecordd, epochrecordd = [], [], [], [] 
    for i in range(len(algorithm)):
        for j in range(len(eta)):
            print('Start training:', algorithm[i], eta[j], layer)
            print('training progress:')
            y, costtrainrecord, costtestrecord, epochrecord = regression(train, traintarget, test, testtarget, layer, act_func, eta[j], beta1, beta2, eps,  algorithm[i], performance, max_epoch)         
            yy.append(y)                                                                                                
            costtrainrecordd.append(costtrainrecord)                                                                
            costtestrecordd.append(costtestrecord)                                                                    
            epochrecordd.append(epochrecord)                                                                           
        plt.subplot(2, 2, i + 1)
        plt.plot(test[1], testtarget[0], label='Test target', color='k', marker='o', linestyle='None',
                 linewidth=4)                                                                                           
        for k in range(len(eta)):
            plt.plot(test[1], yy[k][0], label=f'\u03B7={eta[k]}')                                                      
            plt.title(algorithm[i])
            plt.xlabel('x')
            plt.ylabel('y')
        plt.legend()
        plt.subplot(2, 2, len(algorithm) + i + 1)
        for l in range(len(eta)):
            plt.plot(epochrecordd[l], costtrainrecordd[l], label=f'\u03B7={eta[l]}')                                   
            plt.xlabel('Epoch')
            plt.ylabel('Loss on training set')
            plt.title(algorithm[i])
        plt.legend()
        for m in range(len(eta)):                                                                                       
            print(algorithm[i], eta[m], 'test loss:', costtestrecordd[m][-1])
        yy, costtrainrecordd, costtestrecordd, epochrecordd = [], [], [], []
    plt.show()


def main():
    # === System hyperparameters === #
    algorithm = ['SGD','Adam'] # Optimiser
    performance = 1e-4 # target training MSE loss that will auto stop training
    layer = [3, 1] # Structure
    act_func = ['tanh', 'linear'] # Structure

    # === Algorithm hyperparameters === #
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    #### MODIFY THE CODE HERE ####
    eta = [0.1, 0.01, 0.001, 0.0001] # Learning Rate
    max_epoch = 10000 # Max Epoch

    regression_plot(xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, beta1, beta2, eps, algorithm, performance, max_epoch)


if __name__ == "__main__":
    main()
