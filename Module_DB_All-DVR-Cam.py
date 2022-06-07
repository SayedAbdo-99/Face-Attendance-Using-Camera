import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from DatabaseLayer import addAttendance
import random

#print(random.randint(0,9))

now = datetime.now()
camNumbers=11
path = '/images'
inputVideo = 'input.mp4'
outVideoName = '/Output/video_'+str(random.randint(0,20))+'_'+str(now.date())+'.avi'
images = []
personNames = []

#set persons that named in the images folder
Persons=["HITHM","ZENAB","SAYED","MOHAMED"]
Attendance=[]
Acc = {key: 0 for key in Persons}



myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('db.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

'''
vs = cv2.VideoCapture(inputVideo)
'''
vs=[]
for i in range(camNumbers):    
    vs.append(cv2.VideoCapture(i))
    rtsp = "rtsp://admin2:Zxcvb_12345@192.168.2.200:554/cam/realmonitor?channel="+str(i+1)+"&subtype=1"
    print(str(vs[i]))
    print(len(vs))
    vs[i].open(rtsp)
    vs[i].set(3, 640)  # ID number for width is 3
    vs[i].set(4, 480)  # ID number for height is 480
    vs[i].set(10, 100) 

writer = None
(W, H) = (None, None)
while True:
    #ret,frame=[],[]
    for i in range(len(vs)): #range(camNumbers):    
        ret, frame = vs[i].read()
        print(ret)
        print(frame)
        if ret is False:
            break

        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = faces.shape[:2]
        facesCurrentFrame = face_recognition.face_locations(faces)
        encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print("faceDis: "+str(faceDis))
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personNames[matchIndex].upper()
                print("name: "+name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                attendance(name)
                if name in Persons:
                    if  name not in Attendance:

                        Attendance.append(name)
                print(name+ " attended")
                # check if the video writer is None
                output=cv2.resize(frame, (488, 488))
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    #print((W,H))
                    writer = cv2.VideoWriter(outVideoName, fourcc, 30,(W, H), True)#args["output"]
                # write the output frame to disk
                writer.write(output)
        print(i)
        cv2.imshow('Webcam'+str(i), frame)
        if i<5:
            cv2.moveWindow('Webcam'+str(i), (i*35)*10, 50)
        if i>=5 and i<10:
            cv2.moveWindow('Webcam'+str(i), (i*5)*10, 400)
        if i>=10:
            cv2.moveWindow('Webcam'+str(i), i*10, 800)
            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
if writer is not None:
    writer.release()
for v in vs:
    vs[0].release()
cv2.destroyAllWindows()

delayTime=(now.hour-8)*60+now.minute


for emp in Attendance:
    print("emp: "+str(emp) + "\\n"+"attdate: "+ str(now))
    addAttendance(cam, emp, now)
    print("Attendance Saved")
