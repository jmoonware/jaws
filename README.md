# jaws
Realistic Halloween skeleton jaw automation from audio/video 

See the pdf in the 'doc' folder for an overview and other details.

This project translates a video of someone speaking words into a real-time signal which can drive a servo motor based on the detected jaw position (hence the name of the project, 'jaws')

Here are some step-by-step instructions on how to use:

Best to create a new python venv via:

```
python -m venv jaws-env
```

Then:

- Windows:
```
jaws-env\Scripts\activate
```

- Linux:
```
source jaws-env/bin/activate 
```

Clone the repos via:

```
git clone https://github.com/jmoonware/jaws
git clone https://github.com/jmoonware/jaws_data
git clone https://github.com/jmoonware/jaws_arduino
```

After that install from requirements.txt in the jaws project:

```
pip install -r requirements.txt
```

The basic pipeline is:

- Get a video of a single face saying the words you want the servo motor to mimic
- Run the 'detect_motion.py' script on this video file to generate a motion.txt file
- Strip the audio from the video file using ffmpeg or other tool (see the jaws_data project)
- Use the resulting audio and motion file to generate a header file via the 'generate_header.py' script
- Put the header in the Arduino sketch directory (jaws_arduino/rp2040_jaws) 
- Compile and upoad the sketch to the Arduino RP2040 board

Example using the test data (while in the 'jaws' directory):

```
python detect_motion.py -vf ..\jaws_data\test_raw.mp4
```

If you get bored, hit the 'q' key to stop and save the motion data that got extracted. Should create a 'motion.txt' file in the current 'jaws' directory, about 25 s worth of motion

Next, (optional) extract the audio from the test_raw.mp4 file via

```
ffmpeg.exe -i test_raw.mp4 -map 0:a -ar 20000 -c:a pcm_u8 -ac 1 output20_u8_1.wav
```

Assuming ffmpeg is in your PATH - if not supply the full path in the command above. The output20_u8_1.wav file already exists so this step is optional.

Then, generate the header file via:

```
python generate_header.py -wf ..\jaws_data\output20_u8_1.wav -mf motion.txt -tmax 10
```

The 'tmax' option limits the file to 10s of audio/motion

Then, copy the generated 'jaws_include.h' file to the jaws_arduino\rp2040_jaws directory (a file with this name already exists so overwrite)

Then open the Arduion IDE and compile\upload. Should work!
