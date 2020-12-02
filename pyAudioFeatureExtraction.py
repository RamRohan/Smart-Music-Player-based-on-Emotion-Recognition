import subprocess
x = subprocess.check_output(["python", "audioAnalysis.py", "beatExtraction", "-i", "data/beat/small.wav"], cwd = 'pyaudio_analysis')
print x

y = subprocess.check_output(['python', 'audioAnalysis.py','featureExtractionFile', '-i', '01. Still 24K.mp3','-mw', '1.0','-ms' ,'1.0', '-sw','0.050', '-ss','0.050','-o', 'Therapy.mp3'], cwd = 'pyaudio_analysis')
