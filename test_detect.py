import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

project_path=r'..'
fc=os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_alt.xml')
# Apparently uses this model http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf
# Can download via https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml
facemark_model=os.path.join(project_path,'jaws_data','lbfmodel.yaml')
f=os.path.join(project_path,'jaws_data','test_raw.mp4')

# extraction points of interest from facemarks
# See e.g. https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
POI_Upper=[51,62]
POI_Lower=[57,66]

face_cascade = cv2.CascadeClassifier(fc)

# list of detected faces and facemarks for each frame
faces=[]
facemarks=[]

# create landmark detector and load lbf model:
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(facemark_model)

def detect_and_show(frame):
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)
	# this is where we detect the face area
	# x,y,w,h apparently returned for the face
	faces.append(face_cascade.detectMultiScale(frame_gray))
	if len(faces[-1]) > 0 and len(faces[-1][0])==4:
		# upper left
		ul=(faces[-1][0][0],faces[-1][0][1])
		# bottom right
		br=(ul[0]+faces[-1][0][2],ul[1]+faces[-1][0][3])
		cv2.rectangle(frame, ul, br, (0,255,0))

	# run landmark detector on the faces that we found (usually one):
	ok, landmarks = facemark.fit(frame_gray, faces[-1])
	facemarks.append(landmarks)
	for landmark in landmarks:
		for ip, p in enumerate(landmark[0]):
			c=(0,255,255) # yellow
			if ip in POI_Upper:
				c=(255, 0, 0) # blue
			if ip in POI_Lower:
				c=(0, 0, 255) # red
			cv2.circle(frame, (int(p[0]), int(p[1])), 1, c, 1)

	cv2.imshow('test',frame)

rgb=cv2.VideoCapture(f)	

# Loop over frames and display detection results
while True:
	ret, frame = rgb.read()
	if frame is None:
		print('No frame!')
		break
	
	detect_and_show(frame)
	
	if cv2.waitKey(10)==27: # check escape with 10 ms wait
		break

# now process the results...	
delta_y=[]
idx=[]
for im,fm,face_roi in zip(range(len(facemarks)),facemarks,faces):
	marks=fm[0][0]
	if len(face_roi) > 0:
		df=face_roi[0] # first and should be only detected face
	else:
		continue
	if len(marks)==68:
		nm=np.array(marks)
		yu=np.mean(nm[POI_Upper][:,1])
		yl=np.mean(nm[POI_Lower][:,1])
		xu=np.mean(nm[POI_Upper][:,0])
		xl=np.mean(nm[POI_Lower][:,0])

		# face ROI is x,y,w,h
		if yl>=yu and yu > df[1] and yu < df[1]+df[3] and xu > df[0] and xu < df[0]+df[2]:
			delta_y.append(yl-yu)
			idx.append(im)
		else:
			print("out-of-roi index={0}".format(im))
			
plt.plot(idx,delta_y,marker='o',ms=3)
plt.xlabel("Frame Number")
plt.ylabel("Lip Distance (pixels)")
plt.show()