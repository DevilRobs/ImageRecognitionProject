from django.shortcuts import render
from django.http import HttpResponse
from recognition.services import *

# Create your views here.
def index(request):
    context = {
        "string_message" : irgendwas()
    }
    return render(request, "recognition/index.html", context)
