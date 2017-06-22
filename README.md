# lightnn
The light(\`light\` means not many codes here) deep learning framework for study and for fun. Join us!

## Package structure:

lightnn  
├── __init__.py  
├── __init__.pyc  
├── base  
│   ├── __init__.py  
│   ├── __init__.pyc  
│   ├── activations.py  
│   ├── activations.pyc  
│   ├── initializers.py  
│   ├── initializers.pyc  
│   ├── losses.py  
│   ├── losses.pyc  
│   ├── optimizers.py  
│   └── optimizers.pyc  
├── examples  
│   ├── __init__.py  
│   ├── cnn.py  
│   ├── nn.py  
│   └── test_model.py  
├── layers  
│   ├── __init__.py  
│   ├── __init__.pyc  
│   ├── convolutional.py  
│   ├── convolutional.pyc  
│   ├── core.py  
│   ├── core.pyc  
│   ├── layer.py  
│   ├── layer.pyc  
│   ├── pooling.py  
│   ├── pooling.pyc  
│   └── recurrent.py  
├── models  
│   ├── NeuralNetwork.py  
│   ├── __init__.py  
│   ├── __init__.pyc  
│   ├── models.py  
│   └── models.pyc  
├── ops.py  
└── ops.pyc  

## Modual structure:

### activations

* identity
* sigmoid
* relu
* softmax

### losses

* MeanSquareLoss
* BinaryCategoryLoss
* LogLikelihoodLoss

### initializers

* xavier uniform initializer(glorot uniform initializer)
* default weight initializer
* large weight initializer

### optimizers

* SGD

### layers

* FullyConnected(Dense)
* Softmax
* Dropout
* Flatten



For details, please visit [skyhigh233](skyhigh233.com).
