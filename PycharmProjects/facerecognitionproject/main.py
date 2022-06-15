import cv2
import numpy as np
import face_recognition

imgTonet = face_recognition.load_image_file('ImagesBasic/Tonet.jpg')
imgTonet = cv2.cvtColor(imgTonet,cv2.COLOR_BGR2RGB)
imgtonets = face_recognition.load_image_file('ImagesBasic/joy.jpg')
imgTest = cv2.cvtColor(imgtonets, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTonet)[0]
encodeTonet = face_recognition.face_encodings(imgTonet)[0]
cv2.rectangle(imgTonet,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255),2)

faceLoctonets = face_recognition.face_locations(imgtonets)[0]
encodetonets = face_recognition.face_encodings(imgtonets)[0]
cv2.rectangle(imgtonets,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255),2)

results = face_recognition.compare_faces([encodeTonet],encodetonets)
faceDis = face_recognition.face_distance([encodeTonet],encodetonets)
print(results,faceDis)
cv2.putText(imgtonets,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Antonette Mendiola',imgTonet)
cv2.imshow('Antonette Test',imgtonets)
cv2.waitKey(0)

