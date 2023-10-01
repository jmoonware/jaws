import cv2
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
#import keyboard

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

parser.add_argument('-ov','--output_video', help="Annotated .mp4 video file",default='',required=False)


command_line_args=vars(parser.parse_args(sys.argv[1:]))
video_file=command_line_args['video_file']
output_file=command_line_args['output_file']
output_video_file = command_line_args['output_video']
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
POI_JAW = [7,8,9] # Points 8,9,10 span lower three jawline points
POI_NOSE = [30,32,33,34] # top of both eyes

face_cascade = cv2.CascadeClassifier(face_classifier)

# list of detected faces and facemarks for each frame
faces=[]
facemarks=[]
max_face=[]

# create landmark detector and load lbf model:
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(facemark_model)

def detect_and_show(frame):
	global last_landmarks
	global num_exceptions
	global num_frames
	global frame_rate

	num_frames+=1
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)
	# this is where we detect the face area
	# x,y,w,h apparently returned for the face
	faces.append(face_cascade.detectMultiScale(frame_gray)) # can be multiple
	# always take the largest face detected
	face_sizes = []
	for f in faces[-1]:
		face_sizes.append(f[2]*f[3])
	if len(face_sizes)==0: # nothing detected
		facemarks.append(last_landmarks)
		return(frame)
	max_face.append(np.argmax(face_sizes))
	if len(faces[-1]) > max_face[-1] and len(faces[-1][0])==4:
		# upper left
		ul=(faces[-1][max_face[-1]][0],faces[-1][max_face[-1]][1])
		# bottom right
		br=(ul[0]+faces[-1][max_face[-1]][2],ul[1]+faces[-1][max_face[-1]][3])
		cv2.rectangle(frame, ul, br, (0,255,0),2)

	# run landmark detector on the faces that we found (usually one):

	try:
		ok, landmarks = facemark.fit(frame_gray, faces[-1])
#		print(len(landmarks),max_face)
		facemarks.append([landmarks[max_face[-1]]])
		last_landmarks=[landmarks[max_face[-1]]]
	except Exception as e:
		num_exceptions+=1
		facemarks.append(last_landmarks)
		print("*** Exception {0} at frame {1}".format(num_exceptions,num_frames))
		print(e)
	
	# if we haven't detected any landmarks yet, then bail
	if not last_landmarks:
		return
	for landmark in last_landmarks:
		for ip, p in enumerate(landmark[0]):
			c=(0,255,255) # yellow
			if ip in POI_Upper:
				c=(255, 0, 0) # blue
			if ip in POI_Lower:
				c=(0, 0, 255) # red
			if ip in POI_JAW or ip in POI_NOSE:
				c=(0, 128, 255) # orange-ish
			cv2.circle(frame, (int(p[0]), int(p[1])), 3, c, -1)

	annotated_frame = cv2.putText(frame, "{0}:  {1:.2f}".format(num_frames,num_frames/frame_rate), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,0), 3, cv2.LINE_AA)

	cv2.imshow('output',annotated_frame)
	return annotated_frame

# here is where the main loop starts

if not os.path.isfile(video_file):
	sys.stderr.write("Can't open video file {0} - Goodbye\n".format(video_file))
	sys.exit(1)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	
rgb=cv2.VideoCapture(video_file)	
frame_rate=rgb.get(cv2.CAP_PROP_FPS)
frame_count=rgb.get(cv2.CAP_PROP_FRAME_COUNT)
f_width  = rgb.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
f_height = rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
aspect = f_width/f_height

resize_h = 500
resize_w = resize_h*aspect

cv2.resizeWindow("output",int(resize_w),int(resize_h)) 

print("Frame Rate = {0:.4f} Hz, Frame Count = {1}, T = {2:.3f} s".format(frame_rate,frame_count,frame_count/frame_rate))
print("Frame Width = {0:.0f} pix, Frame Height = {1:.0f}".format(f_width,f_height))

if frame_end < 0:
	frame_end=int(frame_count)
elif frame_end > frame_count:
	frame_end=int(frame_count)

import platform
four_cc = None
recorder=None

if len(output_video_file) > 0:
	if platform.system() == "Windows":
		four_cc = cv2.VideoWriter_fourcc(*"mp4v")
	if four_cc: 
		recorder = cv2.VideoWriter(output_video_file, four_cc, frame_rate, (int(f_width), int(f_height)))

# Loop over frames and display detection results
for fn in range(frame_start,frame_end):
	ret, frame = rgb.read()
	if frame is None:
		print('No frame at frame number {0} of {1}!'.format(fn,frame_count))
		break
	
	annotated_frame = detect_and_show(frame)
	
	if recorder:
		recorder.write(annotated_frame)
	
#	if keyboard.is_pressed('q'):
#		break
		
	if cv2.waitKey(10)==27: # check escape with 10 ms wait
		break

# now process the results...	
delta_y=[]
delta_y_jawline=[]
idx=[]
idx_jawline=[]
print("Processing {0} facemarks, {1} faces, len max_faces {2}".format(len(facemarks),len(faces),len(max_face)))
for im,fm,face_roi,mf in zip(range(len(facemarks)),facemarks,faces,max_face):
	marks=fm[0][0]
	if len(face_roi) > mf:
		df=face_roi[mf] # largest detected face
	else:
		continue
	if len(marks)==68:
		nm=np.array(marks)
		yu=np.mean(nm[POI_Upper][:,1])
		yl=np.mean(nm[POI_Lower][:,1])
		xu=np.mean(nm[POI_Upper][:,0])
		xl=np.mean(nm[POI_Lower][:,0])

		yu_jawline=np.mean(nm[POI_NOSE][:,1])
		yl_jawline=np.mean(nm[POI_JAW][:,1])
		xu_jawline=np.mean(nm[POI_NOSE][:,0])
		xl_jawline=np.mean(nm[POI_JAW][:,0])


		# face ROI is x,y,w,h
		if yl>=yu and yu > df[1] and yu < df[1]+df[3] and xu > df[0] and xu < df[0]+df[2]:
			delta_y.append(yl-yu)
			idx.append(im)
		else:
			print("out-of-roi index={0}".format(im))

		# FIXME: make distance features an array
		if yl_jawline>=yu_jawline and yu_jawline > df[1] and yu_jawline < df[1]+df[3] and xu_jawline > df[0] and xu_jawline < df[0]+df[2]:
			delta_y_jawline.append(yl_jawline-yu_jawline)
			idx_jawline.append(im)
		else:
			print("Jawline: out-of-roi index={0}".format(im))

if show_plots:
	plt.plot(idx,delta_y,marker='o',ms=3,color='red',label='LIPS')
	plt.plot(idx_jawline,delta_y_jawline,marker='o',ms=3,color='blue',label='JAW')
	plt.xlabel("Frame Number")
	plt.ylabel("Distance (pixels)")
	plt.legend()
	plt.show()

# save to file - time vs pixel distance; pixel distance gets renormalized in next step of process
with open(output_file,'w') as f:
	for x,y in zip(idx,delta_y):
		f.write("{0:.6f}\t{1:.1f}\n".format((x+frame_start)/frame_rate,y))

# jawline file
jlf_name = os.path.splitext(output_file)[0]+"_jawline.txt"
with open(jlf_name,'w') as f:
	for x,y in zip(idx_jawline,delta_y_jawline):
		f.write("{0:.6f}\t{1:.1f}\n".format((x+frame_start)/frame_rate,y))

if len(delta_y) == len(delta_y_jawline):		
	filtered_name = os.path.splitext(output_file)[0]+"_filtered.txt"
	delta_y=np.array(delta_y)
	delta_y_jawline=np.array(delta_y_jawline)
	avedat = delta_y-np.mean(delta_y)+delta_y_jawline-np.mean(delta_y_jawline)
	# slight LPF with convolution
	cavdat = np.convolve(avedat,np.array([0.1,0.25,0.5,0.25,0.1]),mode='same')
	with open(filtered_name,'w') as f:
		for x,y in zip(idx_jawline,cavdat):
			f.write("{0:.6f}\t{1:.1f}\n".format((x+frame_start)/frame_rate,y))	
else:
	print("Can't created filtered file - size mismatch")


