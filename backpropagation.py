import numpy as np

"""### Helper functions"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""### Inital data"""

X = np.array([[3.,1.]])
T = np.array([[1.,0.]])


W = np.array([[6.,-3.],[-2.,5.]])
V = np.array([[1.,-2.],[0.25,2.]])

# 1*2 and 2*2 => 1*2
H_in = np.dot(X,W);
# print(H_in)
H = sigmoid(H_in)#output of hidden layer
# print(H)


Y_in = np.dot(H,V)
Y = sigmoid(Y_in)
print(Y)

"""### <u>Loss Function</u>"""

E = Y - T
print(E)

E_sq = E**2
print(E_sq)

L = np.sum(E_sq,axis=1,keepdims=True)
print(L)

"""### <u>Gradients before and after the output sigmoid function</u>"""

grad_Y = 2*E #as per the loss function
grad_Y_in = (Y*(1-Y))*grad_Y
print(grad_Y_in)# 1*2

"""### <u>For a node , you have to perform differentiation for all the incoming weights/connections</u>"""

# d(y_in)/d(w1,w2,...) for a neuron in previous layer to neurons in the next layer
#gives constants of values after activation_function in previous(hidden) layer
grad_V = np.dot(H.T, grad_Y_in)
print(grad_V)
#first column has all result for output node y1
#second column has all result for output node y2



grad_H = np.dot(grad_Y_in,V.T)
print(grad_H)
#rows*cols => output*input

grad_H_in = (H*(1-H))*grad_H
grad_W = np.dot(X.T,grad_H_in)
grad_X = np.dot(grad_H_in,W.T)
print(grad_W)
print(grad_X)

"""### <u>TEST</u>"""

import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
vs = np.sum(arr,axis=0)#sum up items for each column along row direction
hs = np.sum(arr,axis=1)#sum up items for each row along column direction
print(vs.ndim,vs.size,vs)
print(hs.ndim,hs.size,hs)
