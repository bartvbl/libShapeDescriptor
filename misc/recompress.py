import os
import os.path
import zipfile
import lzma
import subprocess
import time

inDir = '/home/bart/Datasets/SHREC2017_QUICCI_IMAGES/'
outdir = '/media/bart/BOOT/SHREC'
compressor = '/mnt/a666854b-88ec-4fb7-9cc5-c167acbd5e9c/home/bart/git/QuasiSpinImageVerification/cmake-build-debug/libSpinImage/compressor'

allfiles = [f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, f)) and f.endswith('.lz')]

processlist = []

for fileindex, file in enumerate(allfiles):
	if fileindex < 0:
		continue
	print("Started", fileindex+1, '/', len(allfiles), ':', file)
	infile = os.path.join(inDir, file)
	tempfile = os.path.join(inDir, '..', file.replace('.lz', ''))
	outfile = os.path.join(outdir, file)
	process = subprocess.Popen(['cat ' + infile + ' | p7zip -d > ' + tempfile + ' && ' + compressor + ' --input="' + tempfile + '" --output="' + outfile + '" --compress && rm ' + tempfile], shell=True)
	processlist.append(process)
	while len(processlist) == 10:
		processEnded = False
		for proc in processlist:
			if proc.poll() != None:
				processlist.remove(proc)
				processEnded = True
		if not processEnded:
			time.sleep(0.1)

for proc in processlist:
	proc.wait()
