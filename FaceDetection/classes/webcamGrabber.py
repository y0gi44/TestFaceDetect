#!/usr/bin/python
import base64
import math
import os
import os.path
import re
from sys import stdin
import sys
import urllib2

import cv2


import numpy as np


class webcamGrabber:

    user = ""
    pwd = ""
    url = ""
    auth=False
    stream= ''
    bytes = ''
    
    def __init__(self, url, usr = '', pwd = ''):
        self.user = usr
        if (len(usr) > 0 ):
            print '[webcamGrabber] Initialisation avec authentification'
            self.pwd = pwd
            self.url = url
            self.auth=True
            
    def __del__(self):
        self.stream.close()
        print '[webcamGrabber] Delete Object webcamGrabber'
    
    
    def connect(self):
        print "[webcamGrabber] Connexion a " + self.url
        request = urllib2.Request(self.url)

        if (self.auth):
            print "[webcamGrabber] Ajout de l'authentification"
            base64string = base64.encodestring('%s:%s' % (self.user, self.pwd)).replace('\n', '')
            request.add_header("Authorization", "Basic %s" % base64string)  
        
        self.stream = urllib2.urlopen(request)
        print '[webcamGrabber] Connexion OK '
        
        
    def read (self):
        while True:
            self.bytes+= self.stream.read(1024)
            a = self.bytes.find('\xff\xd8')
            b = self.bytes.find('\xff\xd9')
            if a!=-1 and b!=-1:
                jpg = self.bytes[a:b+2]
                self.bytes= self.bytes[b+2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
                return  i

