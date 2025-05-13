import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import set_random_seed as keras_set_random_seed
from keras.optimizers import SGD
from keras.initializers import RandomUniform
import matplotlib.pyplot as plt

# Set a global random seed for reproducibility
keras_set_random_seed(64)

# Load the Iris dataset from a .txt file
data = pd.read_csv('IrisData.txt', delimiter=',', header=None)

# Assign column names
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Map iris types (species) to numerical values and shuffle the data
data['species'] = LabelEncoder().fit_transform(data['species'])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values # Select all columns except the last as features
y = data.iloc[:, -1].values # Select the last column as the target (species)
y = to_categorical(y) # Convert the target into one-hot encoded format for classification

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a function to create, train a model, and compute confusion matrix
def train_and_evaluate(input_dim, architecture, activation_functions, optimizer, epochs, batch_size, X_train, y_train, X_val, y_val):
    # Build the model
    model = Sequential()
    for i, (neurons, activation) in enumerate(zip(architecture, activation_functions)):
        if i == 0:
            # Add the input layer and the first hidden layer
            model.add(Dense(neurons, input_dim=input_dim, activation=activation, kernel_initializer=RandomUniform()))
        else:
            # Add subsequent hidden layers or the output layer
            model.add(Dense(neurons, activation=activation, kernel_initializer=RandomUniform()))

    # Compile the model with the specified optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    # Fit the model using the training data and validate using the validation set
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    # Generate predictions for the validation set
    y_pred = model.predict(X_val)
    # Convert predictions and true labels from one-hot encoding to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Compute the confusion matrix for the validation set
    cm = confusion_matrix(y_true, y_pred_classes)

    return history, cm

# -------------------------------- Test 1: Adding / Removing Nodes (Uncomment to run) ----------------------------------
# 4-2-2-3 network
plot_name1 = "4-2-2-3"                                                                                                  
history1, cm1 = train_and_evaluate(
    input_dim=4,                                                                                                    
    architecture=[2, 2, 3],                                                                                             
    activation_functions=['tanh', 'tanh', 'linear'],                                                                   
    optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                  
    epochs=200,                                                                                                        
    batch_size=1,                                                                                                       
    X_train=X_train,                                                                                                    
    y_train=y_train,                                                                                                   
    X_val=X_val,                                                                                                        
    y_val=y_val                                                                                                        
)

keras_set_random_seed(64) # Refresh seed

# 4-7-5-3 network
plot_name2 = "4-7-5-3"                                                                                                  
history2, cm2 = train_and_evaluate(
    input_dim=4,                                                                                                        
    architecture=[7, 5, 3],                                                                                           
    activation_functions=['tanh', 'tanh', 'linear'],                                                                    
    optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                    
    epochs=200,                                                                                                         
    batch_size=1,                                                                                                       
    X_train=X_train,                                                                                                    
    y_train=y_train,                                                                                                   
    X_val=X_val,                                                                                                        
    y_val=y_val                                                                                                        
)

# -------------------------------- Test 2: Adding / Removing Layers (Uncomment to run) ---------------------------------
# # 4-5-3-3-3 network
# plot_name1 = "4-5-3-3-3"                                                                                                # Text that will be added between brackets during plotting
# history1, cm1 = train_and_evaluate(
#     input_dim=4,                                                                                                        # Size of input layer
#     architecture=[5, 3, 3, 3],                                                                                          # Hidden and output layers
#     activation_functions=['tanh', 'tanh', 'tanh', 'linear'],                                                            # Hidden and output layer activation functions
#     optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                    # Optimiser
#     epochs=200,                                                                                                         # Max Epoch
#     batch_size=1,                                                                                                       # Batch Size
#     X_train=X_train,                                                                                                    # Training data
#     y_train=y_train,                                                                                                    # Training data target
#     X_val=X_val,                                                                                                        # Validation data
#     y_val=y_val                                                                                                         # Validation data target
# )
#
# keras_set_random_seed(64)                                                                                               # Refresh seed
#
# # 4-3-3 network
# plot_name2 = "4-3-3"                                                                                                    # Text that will be added between brackets during plotting
# history2, cm2 = train_and_evaluate(
#     input_dim=4,                                                                                                        # Size of input layer
#     architecture=[3, 3],                                                                                                # Hidden and output layers
#     activation_functions=['tanh', 'linear'],                                                                            # Hidden and output layer activation functions
#     optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                    # Optimiser
#     epochs=200,                                                                                                         # Max Epoch
#     batch_size=1,                                                                                                       # Batch Size
#     X_train=X_train,                                                                                                    # Training data
#     y_train=y_train,                                                                                                    # Training data target
#     X_val=X_val,                                                                                                        # Validation data
#     y_val=y_val                                                                                                         # Validation data target
# )

# ----------------------------- Test 3: Changing activation functions (Uncomment to run) -------------------------------
# # Output layer to Softmax
# plot_name1 = "Softmax Output"                                                                                           # Text that will be added between brackets during plotting
# history1, cm1 = train_and_evaluate(
#     input_dim=4,                                                                                                        # Size of input layer
#     architecture=[5, 3, 3],                                                                                             # Hidden and output layers
#     activation_functions=['tanh', 'tanh', 'softmax'],                                                                   # Hidden and output layer activation functions
#     optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                    # Optimiser
#     epochs=200,                                                                                                         # Max Epoch
#     batch_size=1,                                                                                                       # Batch Size
#     X_train=X_train,                                                                                                    # Training data
#     y_train=y_train,                                                                                                    # Training data target
#     X_val=X_val,                                                                                                        # Validation data
#     y_val=y_val                                                                                                         # Validation data target
# )
#
# keras_set_random_seed(64)                                                                                               # Refresh seed
#
# # hidden layers to ReLU
# plot_name2 = "ReLU Hidden"                                                                                              # Text that will be added between brackets during plotting
# history2, cm2 = train_and_evaluate(
#     input_dim=4,                                                                                                        # Size of input layer
#     architecture=[5, 3, 3],                                                                                             # Hidden and output layers
#     activation_functions=[ 'relu', 'relu', 'linear'],                                                                   # Hidden and output layer activation functions
#     optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),                                                    # Optimiser
#     epochs=200,                                                                                                         # Max Epoch
#     batch_size=1,                                                                                                       # Batch Size
#     X_train=X_train,                                                                                                    # Training data
#     y_train=y_train,                                                                                                    # Training data target
#     X_val=X_val,                                                                                                        # Validation data
#     y_val=y_val                                                                                                         # Validation data target
# )


# -------------------------------------------------- Plotting ----------------------------------------------------------
# Configure plotting fontsize
plt.rcParams['font.size'] = 20                                                                                          
plt.rcParams['xtick.labelsize'] = 15                                                                                    
plt.rcParams['ytick.labelsize'] = 15                                                                                    
plt.rcParams['lines.linewidth'] = 2                                                                                    

# Plot training and validation MSE loss curves for both networks
plt.figure(figsize=(7, 5))
plt.plot(history1.history['loss'], label=f'Training Loss ({plot_name1})', linestyle='-')
plt.plot(history1.history['val_loss'], label=f'Validation Loss ({plot_name1})', linestyle=(0, (5, 1)))
plt.plot(history2.history['loss'], label=f'Training Loss ({plot_name2})', linestyle='-')
plt.plot(history2.history['val_loss'], label=f'Validation Loss ({plot_name2})', linestyle=(0, (5, 1)))
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('MSE Loss', fontsize=20)
plt.title('Training and Validation MSE Loss', fontsize=25)
plt.xlim(0, 200)
# plt.ylim(0, 0.5)                                                                                                    
plt.legend(fontsize=17)
plt.grid()
plt.show()

# Re-configure axis font size
plt.rcParams['font.size'] = 24                                                                                         
plt.rcParams['xtick.labelsize'] = 20                                                                                    
plt.rcParams['ytick.labelsize'] = 20    

# Plot confusion matrices for both networks
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['Setosa', 'Versicolor', 'Virginica'])
disp1.plot(cmap='Blues', ax=axes[0], colorbar=False)
axes[0].set_title(f'Confusion Matrix ({plot_name1})', fontsize=25)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Setosa', 'Versicolor', 'Virginica'])
disp2.plot(cmap='Blues', ax=axes[1], colorbar=False)
axes[1].set_title(f'Confusion Matrix ({plot_name2})', fontsize=25)

plt.tight_layout()
plt.show()
