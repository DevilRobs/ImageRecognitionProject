from django.shortcuts import render

from recognition.services import *

# Create your views here.
def index(request):

    irgendwas()

    context = {
        "imageIds": None,
        "locationNames": getLocationNames()
    }
    return render(request, "recognition/index.html", context)

def showPictures(request):

    #Prepare selected location
    location = request.POST["location"].lstrip().rstrip().replace(" ","_").replace("'","_").lower()

    #Check location name
    file = open("recognition//static//recognition//database//poiNameCorrespondences.txt", "r")
    for line in file:
        l = line.split("\t")
        l[1] = l[1].replace("\n", "")
        if location == line.split("\t")[1].replace("\n", ""):
            imageIds = readRankFile(location)
            break
        else:
            imageIds = "No result"

    context = {
        "imageIds": imageIds,
        "location": location,
        "locationNames": getLocationNames()
    }
    return render(request, "recognition/index.html", context)