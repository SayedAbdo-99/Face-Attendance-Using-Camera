# import the necessary packages
from collections import deque
import numpy as np
# import argparse
import pickle
import cv2
import face_recognition
import os

modelPath = 'D:/NEWjob/FaceAttendance/models/CNN_classifier.model'
labelbinPath = 'D:/NEWjob/FaceAttendance/models/CNN_classifier.pickle'
inputVideo = 'D:/NEWjob/FaceAttendance/dataset/Videos/Sayed/02.mp4'
outputVideo = 'D:/NEWjob/FaceAttendance/output'
outVideoName = 'video'


# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faceCascade= cv2.face_cascade.load('haarcascade_frontalface_default.xml')

path = 'C:\\Users\\Injaz Apps 2\\Documents\\Face_Recognition_Project-main\\images'
images = []
personNames = []
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

encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
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


def TestVideo(modelPath=modelPath, labelbinPath=labelbinPath, inputVideo=inputVideo, size=28, outputVideo=outputVideo, outVideoName=outVideoName):
	outputVideo = outputVideo+'\ '+outVideoName+str(size)+'.avi'
	totalNumber = 0
	realNumber = 0
	fakeNumber = 0
	realPrec = 0.0
	fakePrec = 0.0
	# ______________________________________________________________
	# load the trained model and label binarizer from disk
	print("[INFO] loading model and label binarizer...")
	model = load_model(modelPath)  # args["model"]
	lb = pickle.loads(open(labelbinPath, "rb").read())  # args["label_bin"]

	# initialize the image mean for mean subtraction along with the
	# predictions queue
	mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
	Q = deque(maxlen=size)  # args["size"]
	# ______________________________________________________________
	# initialize the video stream, pointer to output video file, and
  	# frame dimensions
	rtsp = "rtsp://admin2:Zxcvb_12345@192.168.1.200:554/cam/realmonitor?channel=10&subtype=1"
	vs = cv2.VideoCapture()
	vs.open(rtsp)
	vs.set(3, 640)  # ID number for width is 3
	vs.set(4, 480)  # ID number for height is 480
	vs.set(10, 100)

	# vs = cv2.VideoCapture(0)#args["input"]
	# fps = vs.get(cv2.CAP_PROP_FPS)
	# print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
	writer = None
	(W, H) = (None, None)
	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		frame = cv2.resize(frame, (488, 488), interpolation=cv2.INTER_AREA)
		cv2.imshow("Orignal", frame)

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		# clone the output frame, then convert it from BGR to RGB
		# ordering, resize the frame to a fixed 224x224, and then
		# perform mean subtraction

		output = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (224, 224)).astype("float32")
		frame -= mean

		facesCurrentFrame = face_recognition.face_locations(frame)
		encodesCurrentFrame = face_recognition.face_encodings( frame, facesCurrentFrame)

		for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
			matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
			print("faceDis: "+faceDis)
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
			print(name+ " attended")

		cv2.imshow('Webcam', frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	vs.release()
	cv2.destroyAllWindows()
		# make predictions on the frame and then update the predictions
		# queue
'''
		preds = model.predict(np.expand_dims(frame, axis=0))[0]
		Q.append(preds)
'''
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# faces = faceCascade.detectMultiScale(gray, 1.3, 4)
		# faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1, minNeighbors=5)

		# Draw a rectangle around the faces
		# for (x, y, w, h) in faces:
		#	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

'''
		# perform prediction averaging over the current history of
		# previous predictions
		results = np.array(Q).mean(axis=0);print(results)
		i = np.argmax(results);print(i)
		label = lb.classes_[i];print(label);print(lb.classes_)
		# draw the activity on the output frame
		text = "{}".format(label).split("_")[0]
		cv2.putText(output, text, (220,40), cv2.FONT_HERSHEY_SIMPLEX,1.25, (255, 255, 0), 3);
		output=cv2.resize(output, (488, 488));
			
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(outputVideo, fourcc, 30,(W, H), True)#args["output"]
				
		# write the output frame to disk
		writer.write(output)
		# show the output image
		cv2.imshow("Output", output)
		if text =='Sayed':
			fakeNumber+=1
		else :
			realNumber+=1



		totalNumber +=1
				
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		
		
	# release the file pointers
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()
	cv2.destroyAllWindows()
	realPrec=(realNumber/totalNumber)*100
	fakePrec=(fakeNumber/totalNumber)*100

	# print("T: "+str(totalNumber)+"\n"+"PN: " + str(posNumber)+ "\n"+"NN: " + str(negNumber)+ "\n"+"posPrec: " + str(posPrec)+ "\n"+"negPrec: " + str(negPrec)+ "\n")
	return totalNumber, realNumber, fakeNumber, realPrec, fakePrec, outputVideo




TestVideo()

'''

'''
def TestYoutubeVideo(videoID,numberFrames):
	yt_video_download_split(videoID,numberFrames);
	TestVideo(modelPath,labelbinPath )
'''


'''
totalNumber, posNumber, negNumber, posPrec, negPrec, outputVideo = TestVideo()
print("T: "+str(totalNumber)+"\n"+"PN: " + str(posNumber)+ "\n"+"NN: " + str(negNumber)+ "\n"+"posPrec: " + str(posPrec)+ "\n"+"negPrec: " + str(negPrec)+ "\n")
'''


















'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())
'''
'''
mPath='models/Fake_Real_classifier.model'
lbPath='models/Fake_Real_classifier.pickle'
inVideo='dataset/UADFV/fake/0002_fake.mp4'
sizeV=1
ov='output/0004_fake_svm'+str(sizeV)+'.avi'

# TestVideo()

'''
