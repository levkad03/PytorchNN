# Pytorch Animal classifier

This is my first convolutional neural network created with **Pytorch**

The test accuracy of this network is **92.03%**!

The NN was trained on [Animal data dataset](https://www.kaggle.com/datasets/likhon148/animal-data)

The NN was created via **Transfer learning**. The base network is [AlexNet](https://en.wikipedia.org/wiki/AlexNet)

To run model locally create conda environment by typing in console

```
conda env create -f environment.yml
```

To train neural network activate conda environment and type ``python train.py``

To run **Tensorboard**, type

```
tensorboard --logdir=runs
```