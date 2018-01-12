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
    return("I am irgendwas!")


def readRankFile(location):

    c = 31

    file = open("recognition//static//recognition//database//poiNameCorrespondences.txt", "r")
    for i, line in enumerate(file):
        l = line.split("\t")
        if l[1][:-1] == location:
            break

        c += 1


    file = open("recognition//ranking", "r")
    images = []
    for line in file:
        words = line.split()
        if words[0] == str(c):
            images.append(words[2])
            if len(images) == 50:
                break

    return images

def getLocationNames():
    locations = []

    file = open("recognition//static//recognition//database//poiNameCorrespondences.txt", "r")
    for i, line in enumerate(file):
        l = line.split("\t")
        locations.append(l[0])

    return locations
