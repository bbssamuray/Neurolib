# Neurolib
An artificial neural network library written from scratch in c++.

At the moment it uses/supports:

- Arbitrary sizes of layers
- Training using backpropagation
- Cross entropy loss function
- Leaky RELU activation function (Sigmoid is experimental)
- Saving and loading models

This repo includes a simple usage sample in main.cpp

You can build it by just running:
```sh
  git clone https://github.com/bbssamuray/Neurolib
  cd Neurolib
  make
```

# Why?
I wrote this purely for the learning experience. This is neither practical or fast.

It doesn't utilize the GPU and it is not multithreaded. Even though it works as expected at the moment, I really doubt it is artchitected well for the task either.

But writing this helped me grasp a lot of the low level concepts I didn't get about neural networks and machine learning.

And I think knowing what is going on under the hood is a vital part of problem solving.

