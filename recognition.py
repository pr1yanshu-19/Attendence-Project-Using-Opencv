import cv2
import numpy as np
import face_recognition

imgElon=face_recognition.load_image_file('Imagesattendence/elon.jpeg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('elontest.jpeg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest=face_recognition.face_locations(imgTest)[0]
encodeElontest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeElon],encodeElontest)
faceDis=face_recognition.face_distance([encodeElon],encodeElontest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


cv2.imshow("Image Elon",imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)
