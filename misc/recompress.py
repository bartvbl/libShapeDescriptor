import os
import os.path
import zipfile
import lzma

directory = '/home/bart/Datasets/SHREC2017_QUICCI_images/'
outdir = '/home/bart/Datasets/SHREC_TEMP/'

allfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.zip')]

for fileindex, file in enumerate(allfiles):
	print(fileindex+1, '/', len(allfiles), ':', file, '                         ', end='\r')
	zip = zipfile.ZipFile(os.path.join(directory, file))
	imagedata = zip.read('quicci_images.dat')
	with lzma.open(os.path.join(outdir, file.replace('.zip', '.lz')), "w") as f:
		f.write(imagedata)
