import numpy as np

# Define the ReLU and sigmoid functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the binary cross entropy loss
def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize weights and biases
W11, W12, W21, W22, W31, W32 = 0.9, 0.4, -1.5, -0.7, -0.2, 1.6
W10, W20, W30 = 0, 0, 0

# Input data point
X1, X2 = 2, -3
y = 1

# Forward pass
z1 = W11 * X1 + W12 * X2 + W10  # Input to neuron 1
h1 = relu(z1)                    # Output of neuron 1
z2 = W21 * X1 + W22 * X2 + W20   # Input to neuron 2
h2 = sigmoid(z2)                 # Output of neuron 2
y_pred = W31 * h1 + W32 * h2 + W30  # Final output before activation

# Output is not passed through an activation function because the last layer is linear
output = y_pred
loss = binary_cross_entropy(y, output)

# Round the output and the loss to the second decimal place
output_rounded = round(output, 2)
loss_rounded = round(loss, 2)

# backward pass 
dL_doutput = -y / output + (1 - y) / (1 - output)
doutput_dW31 = h1
doutput_dW32 = h2
doutput_dW30 = 1
 

dL_dW31 = dL_doutput * doutput_dW31
dL_dW32 = dL_doutput * doutput_dW32
dL_dW30 = dL_doutput * doutput_dW30


dL_dh1 = dL_doutput * W31
dh1_dz1 = 1 if z1 > 0 else 0
dz1_dW11 = X1
dz1_dW12 = X2


dL_dW11 = dL_dh1 * dh1_dz1 * dz1_dW11
dL_dW12 = dL_dh1 * dh1_dz1 * dz1_dW12
print("Backward pass for W11 and W12", dL_dW11, dL_dW12) 
(output_rounded, loss_rounded)
