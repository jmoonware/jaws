import cv2
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import keyboard

project_path=r'..'



last_landmarks=None
num_exceptions=0
num_frames=0

# command line arguments
parser = argparse.ArgumentParser(
                    description = '\n'.join
						(
							['Generates distance between lips of a speaking face',
							'The input video file should contain one face talking',
							'The output file is two columns, time vs. distance in pixels detected'
							]
						),
					epilog = 'The ouput motion file is an input into the generate_header script'
					)

parser.add_argument('-vf','--video_file', help="Name of input .mp4 video file",default=os.path.join(project_path,'jaws_data','test_raw.mp4'),required=True)
parser.add_argument('-o','--output_file', help="Name of output motion file (two column text)",default='motion.txt',required=False)
parser.add_argument('-p','--show_plots',help="If set, plot some stuff to screen",default=False,required=False,action='store_true')

parser.add_argument('-fs','--frame_start',help="If set, frame number to start analyis",default=0,required=False)
parser.add_argument('-fe','--frame_end',help="If set, frame number to end analysis",default=-1,required=False)

command_line_args=vars(parser.parse_args(sys.argv[1:]))
video_file=command_line_args['video_file']
output_file=command_line_args['output_file']
show_plots=bool(command_line_args['show_plots'])
frame_start=int(command_line_args['frame_start'])
frame_end=int(command_line_args['frame_end'])

face_classifier=os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_alt.xml')
# Apparently uses this model http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf
# Can download via https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml
facemark_model=os.path.join(project_path,'jaws_data','lbfmodel.yaml')

# extraction points of interest from facemarks
# See e.g. https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
POI_Upper=[51,62]
POI_Lower=[57,66]

face_cascade = cv2.CascadeClassifier(face_classifier)

# list of detected faces and facemarks for each frame
faces=[]
facemarks=[]

# create landmark detector and load lbf model:
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(facemark_model)

def detect_and_show(frame):
	global last_landmarks
	global num_exceptions
	global num_frames
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
	num_frames+=1
	try:
		ok, landmarks = facemark.fit(frame_gray, faces[-1])
		facemarks.append(landmarks)
		last_landmarks=landmarks
	except Exception as e:
		num_exceptions+=1
		facemarks.append(last_landmarks)
		print("*** Exception {0} at frame {1}".format(num_exceptions,num_frames))
		print(e)
		return

	for landmark in landmarks:
		for ip, p in enumerate(landmark[0]):
			c=(0,255,255) # yellow
			if ip in POI_Upper:
				c=(255, 0, 0) # blue
			if ip in POI_Lower:
				c=(0, 0, 255) # red
			cv2.circle(frame, (int(p[0]), int(p[1])), 1, c, 1)

	cv2.imshow('test',frame)

# here is where the main loop starts

if not os.path.isfile(video_file):
	sys.stderr.write("Can't open video file {0} - Goodbye\n".format(video_file))
	sys.exit(1)
	
rgb=cv2.VideoCapture(video_file)	
frame_rate=rgb.get(cv2.CAP_PROP_FPS)
frame_count=rgb.get(cv2.CAP_PROP_FRAME_COUNT)

if frame_end < 0:
	frame_end=int(frame_count)
elif frame_end > frame_count:
	frame_end=int(frame_count)

# Loop over frames and display detection results
for fn in range(frame_start,frame_end):
	ret, frame = rgb.read()
	if frame is None:
		print('No frame at frame number {0} of {1}!'.format(fn,frame_count))
		break
	
	detect_and_show(frame)
	
	if keyboard.is_pressed('q'):
		break
		
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

if show_plots:
	plt.plot(idx,delta_y,marker='o',ms=3)
	plt.xlabel("Frame Number")
	plt.ylabel("Lip Distance (pixels)")
	plt.show()

# save to file - time vs pixel distance; pixel distance gets renormalized in next step of process
with open(output_file,'w') as f:
	for x,y in zip(idx,delta_y):
		f.write("{0:.6f}\t{1:.1f}\n".format((x+frame_start)/frame_rate,y))

