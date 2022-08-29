# Generic Neural Network

This is a Generic Neural Network built using numpy and matrices that theoretically can be applied to any python project it is wished to be used in.

# Setup

The two absolutely key modules required are **math** and **numpy**, math is preinstalled with python and numpy can be installed using `pip3 install numpy`.

In order to use v2 with the MNIST data set provided in "samples" the **python-mnist** module must also be installed using `pip3 install python-mnist`.
Then the **matplotlib** module can be installed to output the images for showcasing - if this is not required then you must comment out the `from matplotlib import pyplot as plt` line and the three lines that refer to `plt` in the `make_prediction` function.

If the MNIST data is not to be used the lines at the top of the main.py script referenced with "Comment this out if the MNIST dataset is not to be used" should be commented out.

# Running

To run the network, change any required setup constants in the top of the ``main.py`` file within the area labeled ``Configuration settings``.
Then cd/move into the ``src`` (source) folder and run ``python3 main.py``.

## Mismatching Data will Crash the Program

Please bear in mind that if you feed any kind of mismatching data into the network it will crash with some form of error, as the project does not currently have any top level error handling.
This should be solved soon but in the meantime you may need to be aware.
