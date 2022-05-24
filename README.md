# This is Randopster

It takes a set of topster images, and returns a random album recommendation.

To run, create a conda environment using the following command:

conda create --name randopenv --file env.txt

Activate the environment, and within do:

pip install coverpy

Run with:

python dev_randopster.py

# To Note

CV2 is used to display topster images. To proceed from an image press any key. If the window is closed the program will hang. 

The topster folder contains several types of topster examples:

1. topsters/dev_samples contain samples that are used for improvement of Randopster. 
2. topsters/fail_samples contain samples that are known to fail. 
3. topsters/test_samples contain samples that have not yet been tested.
4. topsters/valid_samples contain samples that work as intented, or sufficiently well, with Randopster. 

By default, Randopster is run with topsters from topsters/valid_samples. 

# TODO

When initializing a new environment one might encounter an issue with CV2. Resolve. 
Find a way to get genre information for increased recommendation specificity. 
