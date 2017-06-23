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

* identity(dense)
* sigmoid
* relu
* softmax
* tanh
* leaky relu
* elu
* selu
* thresholded relu
* softplus
* softsign
* hard sigmoid

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
* Conv2d
* MaxPooling
* AvgPooling
* Softmax
* Dropout
* Flatten



For details, please visit [skyhigh233](http://skyhigh233.com).

## References
1. [Keras](https://github.com/fchollet/keras): a polular deep learning framework based on tensorflow and theano.
2. [NumpyDL](https://github.com/oujago/NumpyDL): a simple deep learning framework with manual-grad, totally written with python and numpy.
3. [paradox](https://github.com/ictxiangxin/paradox): a simple deep learning framework with symbol calculation system. Lightweight for learning and for fun. It's totally written with python and numpy.
4. [Bingtao Han's blogs](https://zybuluo.com/hanbingtao/): easy way to go for deep learning.

