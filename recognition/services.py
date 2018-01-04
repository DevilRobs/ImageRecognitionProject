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
    colorDetection()
    # findPedestrian()
    return("I am irgendwas!")

def initNameCorrespondences():
    global allNames
    ##### read all name associations ###
    with open(divPath + "\\poiNameCorrespondences.txt", "r") as myFile:
        content = myFile.readline()
        content = content.rstrip().replace(" " , "_").split("\t")
        allNames = {content[0]: content[1]}
        content = myFile.readline()
        while content:
            content = content.rstrip().replace(" " , "_").split("\t")
            allNames[content[0]] = content[1]
            content = myFile.readline()

    print("Done with reading the names-list")
    return

# WARNING: this method takes very long because it copies all the pictures from the database - use wisely! (-> only run it the first time after downloading the project)
# NOTE: img_deleted is only for debug-reasons (so we can look at the pictures that get sorted out)
def initGPSFiltering():
    #### filter out according to gps coordinates ####
    # create folder where the deleted and the remaining pictures are put
    if os.path.exists(imgPath + "\\img_deleted"):
        shutil.rmtree(imgPath + "\\img_deleted")
    os.makedirs(imgPath + "\\img_deleted")
    if os.path.exists(imgPath + "\\img_filtered"):
        shutil.rmtree(imgPath + "\\img_filtered")
    os.makedirs(imgPath + "\\img_filtered")

    # filter for every folder/topic
    for name in allNames:
        fileName = name
        tree = ET.parse(divPath + "\\xml\\" + fileName + ".xml")
        root = tree.getroot()
        allLatitudes = []
        allLongitudes = []
        latitudeToPictureId = defaultdict(list)
        longitudeToPictureId = defaultdict(list)

        # create subfolder for each topic
        os.makedirs(imgPath + "\\img_deleted\\" + fileName)
        os.makedirs(imgPath + "\\img_filtered\\" + fileName)

        # copy all pictures from the database
        source = "D:\\div-2014\\devset\\img\\" + fileName # TODO here should be the path to the actual database, but I do not want to upload everything...add your own path to your div here!
        # should be something like source = "D:\\Franzi\\Standardordner\\Desktop\\div-2014\\devset\\img\\" + fileName

        dest = imgPath + "\\img_filtered\\" + fileName
        images = os.listdir(source)
        for image in images:
            shutil.copyfile(source + "\\" + image, dest + "\\" + image)

        # reading all latitudes and longitudes from the xml-file
        for child in root:
            latitude = float(child.get("latitude"))
            longitude = float(child.get("longitude"))
            if latitude != 0 and longitude != 0:
                latitude = round(latitude)
                longitude = round(longitude)
                # TODO maybe round to first decimal place instead, and then use as range +-0.5 instead of +-1 to filter out even more
                allLatitudes.append(latitude)
                allLongitudes.append(longitude)
                latitudeToPictureId[latitude].append(int(child.get("id")))
                longitudeToPictureId[longitude].append(int(child.get("id")))

        if debug:
            print("latitudes: ", allLatitudes)
            print("longitudes: ", allLongitudes)

        # getting the most frequent value (is used as base value for filtering out coordinates which are too far away)
        countLatitude = Counter(allLatitudes)
        countLongitude = Counter(allLongitudes)
        if debug:
            print(countLatitude.most_common())
            print(countLongitude.most_common())
        # we can only sort out if at least one of the coordinates contains more than one value
        if countLatitude.__len__() > 1 or countLongitude.__len__() > 1:
            # if one coordinate only contains 1 value, use only the other one
            if countLatitude.__len__() == 1:
                count = countLongitude
                count2 = countLongitude
            elif countLongitude.__len__() == 1:
                count = countLatitude
                count2 = countLatitude
            else:
                # use longitude or latitude depending on which was more frequent
                if countLatitude.most_common()[0][1] > countLongitude.most_common()[0][1] or \
                        countLatitude.most_common()[0][1] == countLongitude.most_common()[0][1]:
                    # use most frequent latitude for default case
                    count = countLatitude
                    count2 = countLongitude
                else:
                    count = countLongitude
                    count2 = countLatitude

            common_value = -1
            isCount2 = 0
            filterOut = 1
            # check if there is a single most frequent value
            if count.most_common()[0][1] == count.most_common()[1][1]:
                # most frequent value is not unique, check other coordinate:
                if count2.most_common()[0][1] == count2.most_common()[1][1]:
                    filterOut = 0
                    print("ATTENTION: there is no most frequent value for the GPS-coordinates of ", name,
                          ", therefore we cannot filter anything out!")
                else:
                    common_value = count2.most_common()[0][0]
                    isCount2 = 1
            else:
                common_value = count.most_common()[0][0]
            if debug:
                print("the most frequent value is ", common_value)

            # if there is a most frequent value, filter according to it
            if filterOut:
                # remove all pictures which are too far away from this most common GPS-coordinate-pair (e.g. +-5)
                gpsRange = range(common_value - 1, common_value + 1)
                gpsToPictureId = defaultdict(list)
                if (isCount2 and count2 == countLongitude) or ((not isCount2) and count == countLongitude):
                    # delete from longitudes-list
                    gpsToPictureId = longitudeToPictureId
                else:
                    # delete from latitudes-list
                    gpsToPictureId = latitudeToPictureId
                keys = gpsToPictureId.keys()
                for k in keys:
                    if not gpsRange.__contains__(k):
                        if debug:
                            print("deleting all pictures with coordinate ", k)
                        # delete all pictures with this coordinate
                        for id in gpsToPictureId.get(k):
                            if debug:
                                print("deleting id ", id)
                            shutil.move(imgPath + "\\img_filtered\\" + fileName + "\\" + str(id) + ".jpg",
                                        imgPath + "\\img_deleted\\" + fileName + "\\" + str(id) + ".jpg")
        else:
            print("ATTENTION: the GPS-coordinates of ", name,
                  " are too similar, therefore we cannot filter anything out!")

    print("\n Done with filtering out pictures which are too far away")
    return


def findFace():
    divPath = "D:\\div-2014"

    imgPath = divPath + "\\devset\\img"
    # create folder where the deleted and the remaining pictures are put
    if os.path.exists(divPath + "\\devset\\img_deleted"):
        shutil.rmtree(divPath + "\\devset\\img_deleted", ignore_errors=True)
    os.makedirs(divPath + "\\devset\\img_deleted")
    '''
    #not needed yet
    if os.path.exists(divPath + "\\devset\\img_filtered"):
        shutil.rmtree(divPath + "\\devset\\img_filtered", ignore_errors=True)
    os.makedirs(divPath + "\\devset\\img_filtered")
    '''
    folderName = "la_madeleine"  # only look at pictures of la madeleine for checking if it works (you can choose whichever folder you like, later we will loop over all folders)
    # create subfolder for each topic
    os.makedirs(divPath + "\\devset\\img_deleted\\" + folderName)
    # not needed yet
    # os.makedirs(divPath + "\\devset\\img_filtered\\" + folderName)

    face_cascade = "recognition\\haarcascade_frontalface_default.xml"
    eye_cascade = cv2.CascadeClassifier('recognition\\haarcascade_eye.xml')
    directory = imgPath + "\\" + folderName
    count = 0
    for file in os.listdir(directory):
        imagePath = os.path.join(directory, file)

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(face_cascade)

        # Read the image
        image = cv2.imread(imagePath)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            grey,
            scaleFactor=1.02,
            # if very close to 1, unneccessary things are detected, if close to 1.1, faces are not detected anymore ( 465191337.jpg only detected with 1.005)
            minNeighbors=20,  # 5
            minSize=(10, 10),  # 30, 30
            flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if len(faces) > 0:
            print("Found {0} faces in ".format(len(faces)), file)
            count += 1
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_grey = grey[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_grey, 1.01)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                # paint green, face is confirmed
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(divPath + "\\devset\\img_deleted\\" + folderName + "\\" + file + ".jpg", image)
            # cv2.imshow("Faces found in "+filename, image)
    print("Found {0} pictures with faces".format(count))

def colorDetection():
    image = cv2.imread('recognition\\static\\recognition\\images\\never_gets_detected.jpg')
    print(checkForColor("la_madeleine", 2907636079))

    # We have to set the boundaries here... quite difficult
    boundaries = [
        ([132, 50, 80], [206, 185, 175])
    ]

    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        cv2.imshow("images", np.hstack([image, output]))
        cv2.waitKey(0)

def findPedestrian():
    divPath = "D:\\div-2014"

    imgPath = divPath + "\\devset\\img"
    # create folder where the deleted and the remaining pictures are put
    if os.path.exists(divPath + "\\devset\\img_deleted"):
        shutil.rmtree(divPath + "\\devset\\img_deleted", ignore_errors=True)
    os.makedirs(divPath + "\\devset\\img_deleted")
    '''
    #not needed yet
    if os.path.exists(divPath + "\\devset\\img_filtered"):
        shutil.rmtree(divPath + "\\devset\\img_filtered", ignore_errors=True)
    os.makedirs(divPath + "\\devset\\img_filtered")
    '''
    folderName = "la_madeleine"  # only look at pictures of la madeleine for checking if it works (you can choose whichever folder you like, later we will loop over all folders)
    # create subfolder for each topic
    os.makedirs(divPath + "\\devset\\img_deleted\\" + folderName)
    # not needed yet
    # os.makedirs(divPath + "\\devset\\img_filtered\\" + folderName)

    face_cascade = "recognition\\haarcascade_frontalface_default.xml"
    eye_cascade = cv2.CascadeClassifier('recognition\\haarcascade_eye.xml')
    directory = imgPath + "\\" + folderName
    count = 0
    for file in os.listdir(directory):
        imagePath = os.path.join(directory, file)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # human = image[y:y+h, x:x+w] TO CROP THE IMAGE OUT
            colorDetection(image)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)

def checkForColor(name, id):
    # Color Naming Histogram
    # Id, Black, BLue, Brown, Grey, Green, Orange, Pink, Purple, red, white, yellow

    # Read the correct csv
    path = "D:\\div-2014\\devset\\descvis\\descvis\\img"
    with open(path+"\\"+name+" CN.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if int(row[0]) == id:
                if (float(row[4]) + float(row[10])) > 0.2:
                    return "Statue"
                else:
                    return "Human"