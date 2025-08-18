import numpy as np

input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
print(input_features.shape)
input_features

target_output = np.array([[0,1,1,1]])
target_output = target_output.reshape(4,1)
print(target_output.shape)
target_output

weights = np.array([[20.92884587],[20.92884587]])
print(weights.shape)
weights

bias = -10.12637593
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(10000000):

    inputs = input_features

    inp = np.dot(inputs, weights) + bias
    out = sigmoid(inp)

    error = out - target_output

    x = error.sum()
    print(x)

    derror_out = error
    dout_din = sigmoid_der(out)

    deriv = derror_out * dout_din

    inputs = input_features.T
    deriv_final = np.dot(inputs, deriv)

    weights -= lr * deriv_final

    for i in deriv:
        bias -= lr * i

print(weights)
print(bias)