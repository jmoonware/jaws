# jaws
Realistic Halloween skeleton jaw automation from audio/video 

Best to create a new python venv via:

python -m venv jaws-env

Then:

jaws-env\Scripts\activate (Windows)
source jaws-env/bin/activate (Linux)

Clone the repos via:

git clone https://github.com/jmoonware/jaws
git clone https://github.com/jmoonware/jaws_data
git clone https://github.com/jmoonware/jaws_arduino

After that install from requirements.txt in the jaws project:

pip install -r requirements.txt

The basic pipeline is:

- Get a video of a single face saying the words you want the servo motor to mimic
- Run the 'detect_motion.py' script on this video file to generate a motion.txt file
- Strip the audio from the video file using ffmpeg or other tool (see the jaws_data project)
- Use the resulting audio and motion file to generate a header file via the 'generate_header.py' script
- Put the header in the Arduino sketch directory (jaws_arduino/rp2040_jaws) 
- Compile and upoad the sketch to the Arduino RP2040 board



