import numpy as np
import matplotlib.pyplot as plt
from Neural_Network import regression                                                                                  

np.random.seed(33)                                                                                                    

plt.rcParams['axes.grid'] = True                                                                                     
plt.rcParams['font.size'] = 18                                                                                        
plt.rcParams['xtick.labelsize'] = 18                                                                                    
plt.rcParams['ytick.labelsize'] = 18                                                                                 
plt.rcParams['lines.linewidth'] = 2                                                                              


xtrain = np.array([np.arange(-1, 1.05, 0.05)]) # Training input
xtrain = np.array([[x] for x in xtrain.flatten()]) # Reconstruct the array to a column vector where each row represents one input
dtrain = 0.8 * xtrain ** 3 + 0.3 * xtrain ** 2 - 0.4 * xtrain + np.random.normal(0, 0.02, np.shape(xtrain)) # Training target
xtest = np.array([np.arange(-0.97, 1.03, 0.1)]) # Test input
xtest = np.array([[x] for x in xtest.flatten()])
dtest = 0.8 * xtest ** 3 + 0.3 * xtest ** 2 - 0.4 * xtest + np.random.normal(0, 0.02, np.shape(xtest)) # Test target

n = np.shape(xtrain)[0]  # Number of training input
N = np.shape(xtest)[0] # Number of test input
xtrainT = xtrain.T   # Transpose training input so that each column represent one input
xtrain1 = np.vstack((np.ones([1, n]), xtrainT)) # Stack a row of 1s on top of the training input to represent bias
dtrainT = dtrain.T
xtestT = xtest.T # Transpose test input as well
xtest1 = np.vstack((np.ones([1, N]), xtestT)) # Stack a row of 1s on top of the test input to represent bias
dtestT = dtest.T

# Plot regression results
def regression_plot(display_option, title, train, traintarget, test, testtarget, layer,
                    act_func, eta, beta1, beta2, epsAdam, algorithm, performance, max_epoch):                           
    y_list, test_loss_list = [], []                                                                                    
    for i in range(len(layer)):
        y, losstrainrecord, losstestrecord, epochrecord = regression(train, traintarget, test, testtarget, layer[i], act_func[i], eta[0], beta1, beta2, epsAdam, algorithm[0], performance, max_epoch)       
        y_list.append(y)                                                                                      
        test_loss_list.append(losstestrecord[-1])                                                          

    # plot the result
    plt.figure(dpi=150, figsize=(9, 6))
    plt.plot(test[1], testtarget[0], label='Test target', color='k', marker='o', linestyle='None', linewidth=4)        
   # plot results for every configuration
    for i in range(len(layer)):
        # Display appropriately according to the user configurate display_option
        if (display_option == "Layers"):
            layer[i].insert(0, 1)                                                                                       
            plt.plot(test[1], y_list[i][0],  label=f'{layer[i]}')                                                     
            print(f"Final test MSE loss of {layer[i]} is {test_loss_list[i]}")                                      

        elif (display_option == "Hidden-Activation"):
            plt.plot(test[1], y_list[i][0], label=f'{act_func[i][0]}')                                                  
            print(f"Final test MSE loss of {act_func[i][0]} is {test_loss_list[i]}")                                  

        elif (display_option == "Output_Activation"):
            plt.plot(test[1], y_list[i][0], label=f'{act_func[i][1]}')                                                 
            print(f"Final test MSE loss of {act_func[i][1]} is {test_loss_list[i]}")                                   

        plt.title(title, fontsize=26)
        plt.xlabel('x', fontsize=24)
        plt.ylabel('y', fontsize=24)
    plt.xlim((-1, 1))                                                                                                   
    plt.legend()
    plt.show()


def main():
    # === System hyperparameters === #
    algorithm = ['SGD'] # Optimiser
    eta = [0.01] # Learning Rate
    max_epoch = 10000  # Max Epoch
    performance = 1e-5  # target training MSE loss that will auto stop training (Here set to very small value to disable this feature)

    # === Algorithm hyperparameters === #
    """ No need to adjust, because in this task SGD is used """
    beta1 = 0.9
    beta2 = 0.999
    epsAdam = 1e-8

    # === Test 1: Adding / Reducing Nodes === #
    layer = [[1, 1], [2, 1], [3, 1], [5, 1], [15, 1]]
    act_func = [['tanh', 'linear'], ['tanh', 'linear'], ['tanh', 'linear'], ['tanh', 'linear'], ['tanh', 'linear'], ['tanh', 'linear']]
    display_option = "Layers"                                                                                           
    title = "Adding / Reducing Nodes"
    regression_plot(display_option, title, xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, beta1, beta2, epsAdam, algorithm, performance, max_epoch)


    # === Test 2: Adding / Reducing Layers === #
    # layer = [[1], [3, 1], [3, 3, 1], [3, 3, 3, 1], [3, 3, 3, 3, 1]]
    # act_func = [['linear'], ['tanh', 'linear'], ['tanh', 'tanh', 'linear'], ['tanh', 'tanh', 'tanh', 'linear'], ['tanh', 'tanh', 'tanh', 'tanh', 'linear']]
    # display_option = "Layers"                                                                                           
    # title = "Adding / Reducing Layers"
    # regression_plot(display_option, title, xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, betaSGDM, epsAdaGrad,
    #                 betaRMS, epsRMS, beta1, beta2, epsAdam, algorithm, performance, max_epoch)


    # === Test 3: Changing Hidden Layer Activation Function === #
    # max_epoch = 30000
    # layer = [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
    # act_func = [['tanh', 'linear'], ['logsig', 'linear'], ['relu', 'linear'], ['relu_leaky', 'linear'], ['linear', 'linear']]
    # display_option = "Hidden-Activation"                                                                                
    # title = "Changing Hidden Layer Activation Function"
    # regression_plot(display_option, title, xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, betaSGDM, epsAdaGrad,
    #                 betaRMS, epsRMS, beta1, beta2, epsAdam, algorithm, performance, max_epoch)


    # === Test 4: ReLU (hidden layer) on larger network === #
    # max_epoch = 30000
    # layer = [[10, 10, 10, 1]]
    # act_func = [['relu', 'relu', 'relu', 'linear']]
    # display_option = "Layers"                                                                                           
    # title = "ReLU (Hidden) on Larger Network"
    # regression_plot(display_option, title, xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, betaSGDM, epsAdaGrad,
    #                 betaRMS, epsRMS, beta1, beta2, epsAdam, algorithm, performance, max_epoch)


    # === Test 5: Changing Output Layer Activation Function === #
    # max_epoch = 30000
    # layer = [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
    # act_func = [['tanh', 'linear'], ['tanh', 'tanh'], ['tanh', 'logsig'], ['tanh', 'relu'], ['tanh', 'relu_leaky']]
    # display_option = "Output_Activation"                                                                               
    # title = "Changing Output Layer Activation Function"
    # regression_plot(display_option, title, xtrain1, dtrainT, xtest1, dtestT, layer, act_func, eta, betaSGDM, epsAdaGrad,
    #                 betaRMS, epsRMS, beta1, beta2, epsAdam, algorithm, performance, max_epoch)



# Run Main
if __name__ == "__main__":
    main()
