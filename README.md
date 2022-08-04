# UniversalNueralNetwork

This is a Generic Neural Network built using numpy and matricies that theoretically can be applied to any python project it is wished to be used in.

**V1 does not work**
(This was an attempt at creating an OOP nueral network to make it easier to understand and process what's going on for people who aren't used to ai)

**V2 does work**
(This is built using numpy and matricies and is a lot harder to follow but does work)

# How to use

The two absolutely key modules required are **math** and **numpy**, math is preinstalled with python and numpy can be installed using `pip3 install numpy`.

In order to use v2 with the MNIST data set provided in "samples" the **python-mnist** module must also be installed using `pip3 install python-mnist`, then to show the data properly in the testing function the **matplotlib** can be installed to output the images for showcasing - if this is not required then you must comment out the `from matplotlib import pyplot as plt` line and the three lines that refer to `plt` in the `make_prediction` function commented accordingly should.

If the MNIST data is not to be used the lines at the top of the main.py script referenced with "Comment this out if the MNIST dataset is not to be used" should be commented out.

# IMPORTANT

Please bare in mind that if you feed any kind of mismatching data into the network it will crash with some form of error, as the project does not currently have any top level error handling.
This should be solved soon but in the mean time you may need to be aware.
