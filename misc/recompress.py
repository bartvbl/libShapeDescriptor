import os
import os.path
import zipfile
import lzma
import subprocess

directory = '/home/bart/Datasets/SHREC2017_QUICCI_images/'
outdir = '/home/bart/Datasets/SHREC_TEMP/'

allfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.zip')]

processlist = []

for fileindex, file in enumerate(allfiles):
	if fileindex < 3200:
		continue
	print("Started", fileindex+1, '/', len(allfiles), ':', file)
	infile = os.path.join(directory, file)
	outfile = os.path.join(outdir, file.replace('.zip', '.lz'))
	process = subprocess.Popen(['unzip -p "' + infile + '" | p7zip > "' + outfile + '"'], shell=True)
	processlist.append(process)
	if fileindex % 10 == 0:
		for proc in processlist:
			print('Waiting..')
			proc.wait()
		processlist = []
		
	
