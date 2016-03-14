#!/usr/bin/python
import glob
import os
import cv2
import numpy as np

class myFaceBdd:
    
    bddName = ""
    bddDirPath = ""
    
    labels = []
    recognizer = cv2.createLBPHFaceRecognizer()
    #recognizer = cv2.createFisherFaceRecognizer()
    #recognizer = cv2.createEigenFaceRecognizer()
    istrainOK = False;
    hasUpdate = False;
    
    def __init__(self, path, name):
        self.bddName = name
        self.bddDirPath = path
        self.loadMyFaceBdd()
        
    def __del__(self):
        print '[myFaceBdd] Delete Object myFaceBdd'
    
    def getListeFilename(self):
        return self.bddName + '.csv'
    
    def getOpencvDbaseName(self) :
        return self.bddName + '.xml'
    
    def loadListeLabels(self):        
        self.labels  = [line.rstrip('\n') for line in open(self.bddDirPath+'/'+self.getListeFilename())]
    
    def loadOpencvBdd(self):
        if (os.path.isfile(self.bddDirPath+'/'+self.getOpencvDbaseName())):
            self.recognizer.load(self.bddDirPath+'/'+self.getOpencvDbaseName())
        else:
            print ('[myFaceBdd] Chargement impossible, tentative de re aprentissage')
            self.restartLearning()
    
    def loadMyFaceBdd(self): 
        if (os.path.isfile(self.bddDirPath+'/'+self.getListeFilename())):
            print '[myFaceBdd] Chargement de la liste des libelles'
            self.loadListeLabels()
            print '[myFaceBdd] Chargement de la base opencv'
            self.loadOpencvBdd()
            self.istrainOK = True;
        else:
            self.recognizer = cv2.createLBPHFaceRecognizer()
            #self.recognizer = cv2.createEigenFaceRecognizer()

    
    def getListTraining(self):
        image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.bddDirPath) for f in filenames if f.endswith('.jpg')]
        images = []
        # labels will contains the label that is assigned to the image
        lbls = []
        for image_path in image_paths:
            idPerson = int(os.path.split(os.path.split(image_path)[0])[1])
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(gray, 'uint8')
            
            images.append(image)
            lbls.append(idPerson)
        
        return images, lbls
    
    def restartLearning(self):
        imgs, lbls = self.getListTraining()
        self.recognizer.train(imgs, np.array(lbls))
        self.istrainOK = True
        self.hasUpdate = True;
    
    def getIdFromName(self, name):
        return self.labels.index(name)
    
    def getDirForId(self, id):
        return self.bddDirPath + '/' + str(id) 
    
    def getDirForName(self, name):
        return self.bddDirPath + '/' + str(self.labels.index(name))
    
    def addPhoto(self,id, img):
        filename = self.getDirForId(id)
        nextFile = filename+ '/' + str(id) +'_'+str(len(glob.glob(filename + '/*.jpg')))+'.jpg'
        print 'saving <'+nextFile + '>'
        cv2.imwrite(nextFile, img)
    
    def addNewPhoto(self, name, img) :
        
        #si le nom n'existe pas deja, on le rajoute et on cree le repertoire des images pour l'entrainement
        if (name not in self.labels):
            print '[myFaceBdd] Creation du nouveau libelle pour <'+name+'>' 
            self.labels.append(name)
            os.mkdir(self.getDirForName(name))
        
        # recuperation de l'id
        id = self.getIdFromName(name)
        print '[myFaceBdd] Ajout de la photo'
        self.addPhoto(id, img)
        print '[myFaceBdd] Mise a jour du recognizer'
        if (self.istrainOK and False):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.recognizer.update(np.array([gray]), np.array([id]) )
            self.hasUpdate = True;

        else:
            self.restartLearning()
    
    def saveMyFaceBdd(self):
        if (self.hasUpdate):
            # sauvegarde des libelles
            fo = open(self.bddDirPath+'/'+self.getListeFilename(), "wb")
            for i in xrange(0, len(self.labels)):
                fo.write(self.labels[i] + '\n')
            fo.close()
            print "[myFaceBdd] Sauvegarde du mapping OK"
            
            #sauvegarde de la base de donnees opencv
            self.recognizer.save(self.bddDirPath+'/'+self.getOpencvDbaseName())
            print "[myFaceBdd] Sauvegarde cvBdd OK"
            self.hasUpdate = False;


    
    def whoisIt(self, img):
        if (self.istrainOK):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            nbr_predicted, conf = self.recognizer.predict(gray)
            lbl = "Index Out ofBound"
            if (nbr_predicted < len(self.labels)):
                lbl = self.labels[nbr_predicted]
            #print 'PREDICT : nbr='+str(nbr_predicted)+'   nom='+lbl+ '  confidence= '+str(conf)
            #print "{} est reconnu avec une probabilite de {}".format(lbl, conf)
            return lbl, conf
        else:
            return "", 0
    