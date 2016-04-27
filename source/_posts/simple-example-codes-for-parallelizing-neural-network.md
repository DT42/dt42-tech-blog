---
title: Simple example codes for parallelizing neural network
date: 2016-04-27 16:00:56
tags: parallelism, deep-learning, english
---

# Abstract

If you are doing deep learning, one of the most challenging tasks in recent years is to make it run faster.  A very common choice is to compute gradient descent in parallel on the distributed systems using either data parallelism or model parallelism.  This article is meant to demonstrate ways to implement basic parallelism concepts from scratch using a simple model.  It will also show how these parallelism techniques really improve the performance of training a neural network.  If you are looking for more materials about the parallelism techniques, I recommend two easy-to-understand articles written by Tim Dettmers [1][2].

This article focuses on the implementations of data parallelism using a very simple neural network model, so the readers only need to have the basic knowledge of neural network (such as what is neural network).  If you are not yet very familiar with deep learning and the mathematics behind, don’t worry, you can still get most of the ideas mentioned in this article.

The example codes are available on https://github.com/DT42/neural-network-model-manipulations.


# Setup development environment

Note: If you are only interested in reading the example codes instead of making your hand (computer) dirty, you can jump to next section directly.

To execute the example codes, the necessary dependencies are listed below (my OS is Ubuntu desktop 14.04):

1. `virtualenv`: To create a clean testing environment.
1. `numpy`: For matrix operations in CPU mode.
1. `mpi4py`: To split and distribute a dataset to worknodes. 
1. `Theano`: For matrix operations in GPU mode.
1. `CUDA`: To provide GPU computing infrastructure.

Note: `scipy` and `six` will also be installed automatically because of the dependency condition.

## System dependencies

Before installing the dependencies listed above, the following system dependencies are required:

```
$ sudo apt-get update
$ sudo apt-get install python-pip python-dev libopenmpi-dev gfortran libblas-dev liblapack-dev
```

## Dependencies required for running examples

Now, we are ready to install the dependencies of the examples.

```
$ virtualenv -p python2.7 nn-test
$ cd nn-test
$ source bin/activate
(nn-test) $ pip install mpi4py numpy Theano
```

To install CUDA 7.5, you just need to follow the instructions in CUDA toolkit documentation [3][4], or leverage Roelof's installation script [5] which will install CUDA 7.0.


# Read the source code, Luke

This section consists of two parts. First, I will demonstrate how to write a simple neural network model to recognize numbers in the digit images from scratch using MNIST dataset.  Second, I will demonstrate how to improve performance by data parallelism.

To download the example codes:

```
$ git clone https://github.com/DT42/neural-network-model-manipulations.git
```

## Simple neural network

The sample code of this simple neural network model is inspired by Andrew Ng's machine learning course hosted on Coursera [6], and the simplified MNIST training data is copied from the neural network assignment.  If you are interested in the mathematical formulas used in the source code, I encourage you to take this MOOC course.

![Figure 1: Simple neural network structure](https://docs.google.com/uc?id=0B6Zw6hselblgallieW11TnZtd0k) 

Figure 1 is the structure of the neural network used in the example `mnist-nn.py`.  Here is the pseudo code of the example:

```python
trainingset, labels = load_dataset()
model = train(trainingset, labels, learningrate, iteration)
outputs = predict(model, inputs)
precision = check_precision(outputs, labels)
```

In the `train` function, the optimizer is batch gradient descent, and the activation function is sigmoid.

If you set learning rate to 2 and iteration to 1000, cost will be around 0.25 and precision will be around 97%.

## Parallelism: MPI + GPU

To reduce the training time, MPI and GPU computing technologies are used here to implement data parallelism.  The core concept is to split the input dataset, distribute the subsets to the worknodes, and collect the results computed by GPU.  The example of data parallelism with GPU computing is `mnist-nn-data-parallelism.py`.

![Figure 2: Concept of data parallelism on GPU cluster](https://drive.google.com/uc?id=0B6Zw6hselblgTGxPUkJ2ZHNhSFk)

MPI standard defines tons of easy-to-use distributed operations [7].  Figure 3 shows some MPI collective functions used in the example:

![Figure 3: Some of MPI collective functions (copied from MPI standard 3.0)](https://docs.google.com/uc?id=0B6Zw6hselblgSFNoVS1jVEgyalk)

In figure 3, the parallel processes can happen in the same machine or in a cluster.  The broadcast function helps a process share information with the other processes.  All the processes will get a copy of the data assigned by the broadcast function.  The scatter function splits a dataset into several subsets and makes the subsets evenly distributed to a group of processes, while the gather function does the opposite, collects data from a group of processes, and reassembles the collected data into a complete dataset.

Let's see some code snippets in the example (modified slightly for readability):

1. To distribute dataset to worknodes (`Scatter`).

    ```python
    sliced_inputs = np.asarray(np.split(inputs, num_worknodes))
    sliced_labels = np.asarray(np.split(labels, num_worknodes))
    inputs_buf = np.zeros((len(inputs)/num_worknodes, Input_layer_size))
    labels_buf = np.zeros((len(labels)/num_worknodes))
    comm.Scatter(sliced_labels, labels_buf)
    ```

1. To collect training results of worknodes (`Gather`).

    ```python
    cost, (theta1_grad, theta2_grad) = cost_function(
        theta1, theta2,
        Input_layer_size,
        Hidden_layer_size,
        Output_layer_size,
        inputs_buf, labels_buf,
        regular=0)
    theta1_grad_buf = np.asarray([np.zeros_like(theta1_grad)] * num_worknodes)
    comm.Gather(theta1_grad, theta1_grad_buf)
    theta1_grad = functools.reduce(np.add, theta1_grad_buf) / num_worknodes
    ```

1. To synchronize the weights of neural network among worknodes (Bcast).

    ```python
    theta1 -= learningrate * theta1_grad
    comm.Bcast([theta1, MPI.DOUBLE])
    ```

Because there are a lot of matrix multiplications involved in the training process, It is very good to use GPU to speed up the computation.  Theano provides high level functions so you don’t need to worry about the details communicating with GPU.  Here is the source code using Theano to multiply two matrices:

```python
def gpu_matrix_dot():
    x = T.matrix('x')
    y = T.matrix('y')
    z = T.dot(x, y)
    f = theano.function(
        [x, y], z)
    return f
    
Matrix_dot = gpu_matrix_dot()
c = Matrix_dot(a, b)
```

Now, you have seen the most essential concepts of data parallelism.  Here I would like to share more implementation tips:

1. Compile a Theano function once, and use it repeatedly: It is time-consuming to compile a Theano function; your program can be very slow if the compilation is required every time when the  Theano function is called.
1. Check memory ordering of a numpy array carefully: In the examples, we get matrices by `scipy.io.loadmat`, but their memory ordering is Fortran-order instead of C-order [8].  If we broadcast a Fortran-order buffer without notifying this pitfall, we might create C-order buffers to the other worknodes, and they will receive unexpected values in their buffers. For more details, you can check the source codes in `tests/tests_broadcast-weights.py`.

## Environment

A 2-worknode GPU cluster is used to measure the performance of data parallelism.  The hardware information of a worknode is:

* CPU: Intel i7-5930 3.5 GHz
* system memory: 32 GB
* network speed: 1 Gbps Ethernet interface
* GPU: Nvidia Titan X

To measure the network speed between two worknodes, `nc` is a convenient tool to create a TCP connection between a temporary client-server pair used to send testing data and get the connection speed [9]:

```
# temporary server, IP is 192.168.1.1
worknode1 $ nc -vvln 50001 > /dev/null
Listening on [0.0.0.0] (family 0, port 50001)
Connection from [192.168.1.2] port 50001 [tcp/*] accepted (family 2, sport 41018)

# temporary client
worknode2 $ dd if=/dev/zero bs=1M count=1K | nc -vvn 192.168.1.1 50001
Connection to 192.168.1.1 50001 port [tcp/*] succeeded!
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB) copied, 9.11171 s, 118 MB/s
```

## Scenarios

Here are experiment scenarios:

1. Data parallelism vs local execution: with CPU
1. Data parallelism vs local execution: with GPU

The parameters of the learning algorithm:

* learning rate: 0.1
* gradient descent iteration: 60

## Experiment results

Figure {4,5,6} show the results of data parallelism performance comparison, and tells us two essential facts:

1. Data parallelism brings performance improvement.
1. GPU computing seems to be slower than CPU computing, is it true?

![Figure 4: Data parallelism performance comparison](https://drive.google.com/uc?id=0B6Zw6hselblgRkw4eEVjR2t2WDQ)

![Figure 5: Performance comparison at forward propagation stage](https://drive.google.com/uc?id=0B6Zw6hselblgSGdPWV9GMnRfSmc)

![Figure 6: Performance comparison at back propagation stage](https://drive.google.com/uc?id=0B6Zw6hselblgWjBqNGZMRzNFc00)

At the first glance, the improvement seems to be trivial.  If the dataset is split and distributed evenly on two machines (so doubled computing units), of course the speed can also be two times faster than using a single machine alone.  However, the really important fact behind is that we are now able to scale-out the computing ability from one machine to the GPU cluster easily.  Since GPU computing is still one of the most effective technology for deep learning, this technique can make the learning process much faster.

Secondly, the fact that GPU is slower in those figures does not mean that CPU is a better tool than GPU for deep learning; it is also not a bug in the source code (lol).  Actually, it just reflects the fact that the cost of memory copy between system and GPU memories is expensive.  Because the size of the sample neural network is small, the computation loads are not really heavy.  Therefore, the improvement to the total computing time by using GPU is overwhelmed by the time consumed while copying inputs and outputs between system memory and GPU memory.

When the size of the data is bigger, the ratio of time to compute and time to execute memory copy is also getting larger; GPU computing starts to show its power.  Huge dataset and larger number of hidden layers make GPU computing a desirable tool for deep learning.  Let’s make a very simple experiment to test if our understanding about GPU performance is true.  In figure {7,8}, two matrix of different sizes are multiplied.  One can see that GPU is only slower than CPU when one of the matrix sizes is below 120x30 while the size of another one is below 30x20.

![Figure 7: When the data is small, the total time consumed by GPU computing is longer than using CPU alone](https://drive.google.com/uc?id=0B6Zw6hselblgZGlQSE55SEN3OFk)

![Figure 8: GPU computing is far faster than CPU computing on large matrices](https://drive.google.com/uc?id=0B6Zw6hselblgMXhfMExoM0sxeHc)


# Conclusion

For deep learning, GPU cluster is one of the most powerful tools to help reduce the computing time [10].  To overcome the hardware limitation of a single machine, we can also scale out the computing resource to make distributed systems.  Parallelism enables data scientists and engineers to propose more powerful solutions, and get experimental results sooner.

Besides of architecture design of GPU, there are also other interesting technologies which try to reduce the computation time, such as CUDA-Aware MPI and Nvidia GPUDirect [11], specific computing hardware (Teradeep [12], Singular Computing [13]).

Model compression is another interesting topic which reduces the computational loads from a different approach without dropping precision much [14][15].

Special thanks:

 * Tammy Yang's suggestions for grammatical mistakes.


# References

1. [Tim Dettmers, How to Parallelize Deep Learning on GPUs Part 1/2: Data Parallelism, 2014]( http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/)
1. [Tim Dettmers, How to Parallelize Deep Learning on GPUs Part 2/2: Model Parallelism, 2014](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/)
1. [Nvidia, CUDA 7.5 downloads](https://developer.nvidia.com/cuda-downloads)
1. [Nvidia, CUDA toolkit document](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#ubuntu-installation)
1. [Roelof Pieters, II: Running a Deep Learning (Dream) Machine, 2015](http://graphific.github.io/posts/running-a-deep-learning-dream-machine/)
1. [Andrew Ng, Machine Learning, Coursera](https://www.coursera.org/learn/machine-learning/)
1. [MPI Forum, MPI: A Message-Passing Interface Standard Version 3.0, 2012](http://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf)
1. [Numpy.org, Numpy internals](http://docs.scipy.org/doc/numpy/reference/internals.html#multidimensional-array-indexing-order-issues)
1. [How do you test the network speed between two boxes?](http://askubuntu.com/questions/7976/how-do-you-test-the-network-speed-betwen-two-boxes)
1. [Coates, Adam, et al. "Deep learning with COTS HPC systems." Proceedings of the 30th international conference on machine learning. 2013.](http://jmlr.org/proceedings/papers/v28/coates13.pdf)
1. [Jiri Kraus, An Introduction to CUDA-Aware MPI, 2013](https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/)
1. [Teradeep](http://www.teradeep.com/)
1. [Tom Simonite, Why a chip that’s bad at math can help computers tackle harder problems, MIT technology review, 2016](https://www.technologyreview.com/s/601263/why-a-chip-thats-bad-at-math-can-help-computers-tackle-harder-problems/)
1. [Han, Song, et al. "Learning both Weights and Connections for Efficient Neural Network." Advances in Neural Information Processing Systems. 2015.](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)
1. [Han, Song, et al. "EIE: Efficient Inference Engine on Compressed Deep Neural Network." arXiv preprint arXiv:1602.01528 (2016).](http://arxiv.org/pdf/1602.01528v1.pdf)
