# This is Randopster

It takes a set of topster images, and returns a random album recommendation.

To run, create a conda environment using the following command:

`conda create --name randopenv --file env.txt`

Activate the environment, and within do:

`pip install coverpy`

Run with:

`python dev_randopster.py`

# Docker

Randopster has now been dockerized. 

To build, from randopster folder, run:

`sudo docker build -t randop:tag .`

To spin up interactive container from image, run:

`sudo docker run --rm -it randop:tag`

The script used as entrypoint for dockerization is slightly modified from the original due to issues with CV2 within docker. Instead of using cv2.imshow() an attempt is made to use matplotlib.pyplot.imshow(), although this has not successfully been implemented as of yet.

# To Note

pytesseract wrapper depends on Tesseract OCR being installed on the system. In this case located at '/usr/bin/tesseract'.

Tesseract OCR is compatible with multiple languages at once. It is now implemented to recognize both Enlish and Japenese, as these are particularly prevalent in topsters. It comes with a certain cost to the interpretation of either language and additional clean-up measurements that must be taken for handling symbols. To enable more languages all one has to do is to download additional models from https://github.com/tesseract-ocr/tessdata and move them to the local tessdata directory, in this case '/usr/share/tesseract-ocr/5/tessdata/'.  

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

See if size of docker images can be reduced. 

Get dockerized version to display images. 
