#
# Creates header '.h' include file for an Arduino sketch that allows synchronizing audio sound 
#	and motion of hobby level RC servos
#
# The Seeeduino XIAO RP2040 series has 8 16-bit counters called "slices", each with pairs of outputs A and B
# 
# From the RP2040 data sheet, the period of a counter "slice" is:
# T = (TOP+1)*(CSR_PH_CORRECT+1)*(DIV_INT + DIV_FRACT/16)
# The divider is a fixed point 8.4 value, so non-whole integer scale factors are used 
# Of course, the resulting period must be an integer
# Note that the period is TOP+1, not TOP, as the number of counter states in a period is [0,TOP] inclusive
#
# Three timers (slices) are used for syncing audio and motor control pulses:
#    Audio Slice (on D0, D1) - the PWM that generates the audio signal
#    IRQ Slice (Internal) - the audio update frequency counter, generates IRQ's that are serviced by the ISR
#    Motor Slice (D2, D3) - the PWM for controlling the servo
#
# Note that the clock rate of the RP2040 means you can't get full 44 kHz, 16 bit high quality audio using PWM...
# A good trade-off seems to be 8 bit audio at 20 KHz sampling (fine for speech reproduction)
# There is ~2MB of flash on the RP2040, so ~100 s of 8-bit 20kHz audio can be stored
#

import numpy as np
import sys, os
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse

# command line arguments
parser = argparse.ArgumentParser(
						description = '\n'.join(
							['Generates include file for synchronizing sound and motion on an RP2040 Arduino',
							'The input motion file is created using \'detect_motion.py\' script on an MP4 video',
							'\tThe motion file should be two columns, time vs. z position in pixels',
							'The input audio file can be extracted from the video using ffmpeg'
							]
						),
						epilog = 'Remember to include the generated \'.h\' file in your sketch!'
					)

parser.add_argument('-wf','--wav_file', help="Name of input .wav audio file",default='audio.wav',required=False)
parser.add_argument('-mf','--motion_file', help="Name of input motion file",default='motion.wav',required=False)
parser.add_argument('-o','--output_file', help="Name of output .h header file",default='jaws_include.h',required=False)
parser.add_argument('-tp','--test_pattern',help="If set, generate test pattern",default=False,required=False,action='store_true')
parser.add_argument('-p','--show_plots',help="If set, plot some things",default=False,required=False,action='store_true')
parser.add_argument('-tmax','--time_max',help="Maximum time (s) to use",default=3600,required=False)
parser.add_argument('-ma','--motor_amplitude_fraction',help="Fraction of available motor range to use",default=0.75,required=False)
parser.add_argument('-mo','--motor_offset_fraction',help="Fraction of motor for min motion signal",default=0.25,required=False)

command_line_args=vars(parser.parse_args(sys.argv[1:]))
wav_file=command_line_args['wav_file']
motion_file=command_line_args['motion_file']
output_file=command_line_args['output_file']
generate_test_pattern=bool(command_line_args['test_pattern'])
time_max=float(command_line_args['time_max'])
show_plots=bool(command_line_args['show_plots'])

# comments that appear in output header file
msg=[]
msg.append("//")
msg.append("// Jaws header file generator - {0}".format(str(dt.now())))
msg.append("//")

# clock_freq=48000000 # Hz for Seeeduino Xiao SAMD21
clock_freq=133000000 # Hz for Seeeduino Xiao RP2040
audio_bits = 8 # want this many steps in PWM counter for audio
# will get as close as possible but constrained by audio_bits and clock_freq
audio_sample_rate = 20000 # Hz

#
# These are standard servo control values
# The motor moves from 0-180 degrees when pulses of a given length (0.5-2 ms) are sent at 50 Hz 
# 
motor_pwm_freq=50 # Hz
motor_pulse_min_ms = 0.5 # ms
motor_pulse_max_ms = 2 # ms
motor_pwm_bits = 12 # at 8 us dead-band in 1.5 ms range = 187 values, but too low a resolution can cause jittering
motor_update_sampling_rate=50 # Hz, how often to update motor pulse width (anything over 50 Hz is pointless)

motor_amplitude_fraction=float(command_line_args['motor_amplitude_fraction']) # [0,1], 0.5 = 50% p-p range of motion
motor_offset_fraction=float(command_line_args['motor_offset_fraction']) # [0,1], 0.5 = min of motion value is middle of motor range

# calculate a few things for the pwm counters on the microprocessor
# this is the maximum rate we can send audio pwm pulses given the desired amplitude resolution
audio_pwm_freq = clock_freq/(1<<audio_bits)
audio_top = (1<<audio_bits)-1

msg.append("// System Clock Freq = {0} MHz".format(clock_freq*1e-6))
msg.append("//")
msg.append("// Requested Audio Parameters:")
msg.append("//     Resolution = {0} bits, Sample Frequency {1} Hz".format(audio_bits,audio_sample_rate))
msg.append("//     Audio PWM freq (audio pulse rep rate) {0} Hz".format(audio_pwm_freq))
msg.append("//")

# number of audio PWM cycles at this audio sampling rate/resolution
# Each PWM cycle is 2^audio_bits/clock_freq long
# The top count is rounded down to nearest integer, leading to a slightly higher audio sample rate
irq_top=(1<<audio_bits)*int(np.floor(audio_pwm_freq/audio_sample_rate))-1

# Very low audio sampling rates might make this too many clock cycles...
if irq_top >= 2**16:
	sys.stderr.write("*** Requested audio update count at {0} too high - increase audio sampling rate {1} Hz\n".format(irq_top,audio_sample_rate))
	sys.stderr.write("*** Min audio sample rate is {0} Hz without clock div\n".format(clock_freq/((1<<16)+1)))
	sys.exit()

actual_audio_sample_rate=clock_freq/(irq_top+1) # remember [0,TOP] inclusive...

msg.append("// Actual Audio Parameters:")
msg.append("//     Sample Frequency {0} Hz".format(actual_audio_sample_rate))
msg.append("//     Audio IRQ TOP Count {0}".format(irq_top))
msg.append("//     Number of audio PWM cycles per update period {0}".format((irq_top+1)/(1<<audio_bits)))
msg.append("//")

# this is the prescaler the motor pwm - every irq will update the pwm register of the audio
# after a number of irq's that add up to the requested motor update rate, the motor pwm width will be updated as well
motor_clock_period = 1e-3*(motor_pulse_max_ms-motor_pulse_min_ms)/(1<<motor_pwm_bits) # approximate
motor_div=int(clock_freq*motor_clock_period) # actual clock divider value
motor_clock_freq=clock_freq/motor_div # actual
motor_top = int(motor_clock_freq/motor_pwm_freq)-1

if motor_top >= 2**16:
	sys.stderr.write("Requested motor pwm update count at {0} too high - decrease bit resolution\n".format(motor_top,motor_clock_freq))
	sys.exit()

pwm_min_count=int(motor_pulse_min_ms*1e-3*motor_clock_freq)
pwm_max_count=int(motor_pulse_max_ms*1e-3*motor_clock_freq)

msg.append("// PWM range for motor is between {0} and {1} at {2} Hz (prescaled by {3})".format(pwm_min_count,pwm_max_count,motor_clock_freq,motor_div))
msg.append("// Motor range is {0} steps".format(pwm_max_count-pwm_min_count))

# finally, how many irqs should we wait to update motor? 

motor_irq_update_count=(int)(actual_audio_sample_rate/motor_update_sampling_rate)
msg.append("// Update motor position every {0} irq's (actual update rate = {1} Hz)".format(motor_irq_update_count,actual_audio_sample_rate/motor_irq_update_count))

actual_motor_sampling_rate=actual_audio_sample_rate/motor_irq_update_count
msg.append("// Actual Motor Sampling Rate {0} Hz, motor_top={1}".format(actual_motor_sampling_rate,motor_top))

# 
# Note: using 'const' on array forces it to be stored in SPI flash
#
msg.append("#define AUDIO_TOP {0}".format(audio_top))
msg.append("#define IRQ_TOP {0}".format(irq_top))
msg.append("#define MOTOR_TOP {0}".format(motor_top))
msg.append("#define MOTOR_DIV {0}".format(motor_div))
msg.append("#define MOTOR_UPDATE {0}".format(motor_irq_update_count))

# generate test pattern
if generate_test_pattern:
	# test pattern parameters
	# sweeps a tone from min to max freq and back 
	# simultaneously sweeps motor from min to max angle
	msg.append("// Generating test pattern")
	min_freq=50
	max_freq=1000 # Hz
	sweep_period = 0.5 # seconds
	number_of_sweeps = 10 #

	t_half_sweep=np.arange(0,sweep_period,1/actual_audio_sample_rate)
	N_half=len(t_half_sweep)
	idx_half_sweep=np.arange(0,N_half)
	amp=idx_half_sweep/(N_half-1) # modulate signal by this much

	# align min, max to closest sample freq
	min_freq = actual_audio_sample_rate/int(actual_audio_sample_rate/min_freq)
	max_freq = actual_audio_sample_rate/int(actual_audio_sample_rate/max_freq)
	freqs= min_freq+((max_freq-min_freq)/(N_half-1))*idx_half_sweep
	half_sweep=255*(amp)*np.sin(2*np.pi*freqs*t_half_sweep) # modulated sin wave
	full_sweep=np.append(np.array(half_sweep),-np.array(half_sweep[::-1]))
	full_amp = np.append(amp,amp[::-1])
	
	# lists work better here
	repeat_sweeps=[]
	repeat_amps=[]
	for sn in range(number_of_sweeps):
		repeat_sweeps.extend(list(full_sweep))
		repeat_amps.extend(list(full_amp))
		
	t_repeat=np.arange(0,len(repeat_sweeps))/actual_audio_sample_rate
	if show_plots:
		plt.plot(t_repeat,repeat_sweeps)
		plt.show()

	# note that this will probably play back at a slighly different rate (integer sampling freq)
	# OK as this is only a diagnostic
	wavfile.write('test_pattern.wav',int(actual_audio_sample_rate),np.array(repeat_sweeps,dtype=np.int8))

	# already on actual sample time base
	resample_audio_t = t_repeat
	# renormalize to full range audio bits
	slope=((1<<audio_bits)-1)/(np.max(repeat_sweeps)-np.min(repeat_sweeps))
	resample_audio = ((repeat_sweeps-min(repeat_sweeps))*slope)[resample_audio_t <= time_max]
	# resample to actual update freq
	n_resamples=int(len(resample_audio_t)*actual_motor_sampling_rate/actual_audio_sample_rate)
	# these are the output motion values
	resample_motion_t = np.arange(0,n_resamples)/actual_motor_sampling_rate
	motor_offset=motor_offset_fraction*(pwm_max_count-pwm_min_count)+pwm_min_count
	motor_slope=(pwm_max_count-pwm_min_count)*motor_amplitude_fraction
	# resample onto actual time samples and clip if out of range
	remap=motor_slope*np.interp(resample_motion_t,resample_audio_t,repeat_amps)+motor_offset
	clip_max=(remap>pwm_max_count)
	clip_min=(remap<pwm_min_count)
	remap[clip_max]=pwm_max_count
	remap[clip_min]=pwm_min_count
	resample_motion = np.array(remap,dtype=int)[resample_motion_t <= time_max]
	
else:
	msg.append("// Audio file is {0}".format(wav_file))
	msg.append("// Motion file is {0}".format(motion_file))
	msg.append("// Ouput file is {0}".format(output_file))
	
	# attempt to load audio file
	if os.path.isfile(wav_file):
		rate,data=wavfile.read(wav_file)
	else:
		sys.stderr.write("Can't find {0} - end\n".format(wav_file))
		sys.exit(1)

	# resample to exact audio update rate
	# WAV format only supports integer sampling frequencies and the actual sampling freq may not end up being an integer...
	sample_t=np.arange(0,len(data))/rate
	n_resamples=int(len(data)*actual_audio_sample_rate/rate)
	resample_audio_t=np.arange(0,n_resamples)/actual_audio_sample_rate
	resample_audio=np.interp(resample_audio_t,sample_t,data)[resample_audio_t <= time_max]
	
	# load motion file - should be two columns of time vs. pixel position - will renormalize below
	# probably something like 29.xxx Hz
	if os.path.isfile(motion_file):
		with open(motion_file,'r') as f:
			lines=f.readlines()
		motion_data=[]
		motion_t=[]
		for l in lines:
			motion_t.append(float(l.split()[0]))
			motion_data.append(float(l.split()[1]))
		motion_t=np.array(motion_t)
		motion_data=np.array(motion_data)
		# resample to actual update freq
		video_frame_rate=1/(motion_t[1]-motion_t[0])
		n_resamples=int(len(motion_t)*actual_motor_sampling_rate/video_frame_rate)
		# these are the output motion values
		resample_motion_t = np.arange(0,n_resamples)/actual_motor_sampling_rate
		motor_offset=motor_offset_fraction*(pwm_max_count-pwm_min_count)+pwm_min_count
		motor_slope=((pwm_max_count-pwm_min_count)/(np.max(motion_data)-np.min(motion_data)))*motor_amplitude_fraction
		resample_motion = np.interp(resample_motion_t,motion_t,motor_slope*motion_data+motor_offset)[resample_motion_t <=time_max]		
	else:
		sys.stderr.write("Can't find {0} - end\n".format(motion_file))
		sys.exit(1)

# if we got here, have resample_audio and resample_motion values, can output to header file as arrays...

if show_plots:
	fig,ax1=plt.subplots()
	ax1.plot(resample_audio_t[resample_audio_t<=time_max],resample_audio,color='green')
	ax2=ax1.twinx()
	ax2.plot(resample_motion_t[resample_motion_t<=time_max],resample_motion,color='red',marker='o')
	plt.show()

def array_type(bits):
	atype='uint32_t'
	abytes=4
	if bits <=8:
		atype='uint8_t'
		abytes=1
	elif bits >8 and bits <=16:
		atype='uint16_t'
		abytes=2
	return(atype,abytes)
	
with open(output_file,'w') as f:
	for m in msg:
		f.write(m+'\n')
	audio_bytes=array_type(audio_bits)[1]*len(resample_audio)
	f.write("// Audio data = {0} bytes\n".format(audio_bytes))
	motion_bytes=array_type(motor_pwm_bits)[1]*len(resample_motion)
	f.write("// Motion data = {0} bytes\n".format(motion_bytes))
	f.write("// Total Required Data Bytes = {0}\n".format(audio_bytes+motion_bytes))

	f.write("#define AUDIO_SAMPLES {0}\n".format(len(resample_audio)))
	f.write("#define MOTOR_SAMPLES {0}\n".format(len(resample_motion)))


	# audio array
	f.write("\nconst {0} audio_buffer[{1}] = {{\n".format(array_type(audio_bits)[0],len(resample_audio)))
	for v in resample_audio[:-1]:
		f.write('\t'+str(int(v))+',\n')
	f.write('\t'+str(int(resample_audio[-1])))
	f.write("};\n")
	
	# motion array
	f.write("\nconst {0} motor_buffer[{1}] = {{\n".format(array_type(motor_pwm_bits)[0], len(resample_motion)))
	for v in resample_motion[:-1]:
		f.write('\t'+str(int(v))+',\n')
	f.write('\t'+str(int(resample_motion[-1])))
	f.write("};\n")

