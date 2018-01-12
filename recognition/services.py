#/recognition
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import os
import shutil
import cv2
import sys
import csv

import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

debug = 0
divPath = os.getcwd() + "\\recognition\\static\\recognition\\database"
imgPath = os.getcwd() + "\\recognition\\static\\recognition\\images"
allNames = []

# Put your code here!!!!
def irgendwas():
    # initNameCorrespondences()
    # only run this method for the first time (so that you get the filtered images), you can then comment it out again
    # initGPSFiltering()
    # findFace()
    # colorDetection()
    # findPedestrian()
    return("I am irgendwas!")


def readRankFile():
    file = open("recognition\\ranking", "r")
    images = []
    for line in file:
        words = line.split()
        images.append(words[2])

    return images

def getLocationNames():
    locations = []

    file = open("recognition\\static\\recognition\\database\\poiNameCorrespondences.txt", "r")
    for line in file:
        l = line.split("\t")
        locations.append(l[0])

    return locations