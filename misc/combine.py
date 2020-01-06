from os import listdir
from os.path import isfile, join
import json
import numpy as np

dumpdir = '../PIXEL_COUNTS'

pixelfiles = [f for f in listdir(dumpdir) if isfile(join(dumpdir, f))]

print(len(pixelfiles))

increasingCounts = np.zeros(64*64, dtype=np.uint64)
decreasingCounts = np.zeros(64*64, dtype=np.uint64)
totalImageCount = np.zeros(1, dtype=np.uint64)

for index, file in enumerate(pixelfiles):
	print('\r%i/%i: %s' % (index+1, len(pixelfiles), file), end='')
	with open(join(dumpdir, file), 'r') as infile:
		file_contents = json.load(infile)
		for i in range(0, 64*64):
			increasingCounts[i] += file_contents['increasingCounts'][i]
			decreasingCounts[i] += file_contents['decreasingCounts'][i]
		totalImageCount[0] += file_contents['imageCount']

print()
print('Combined pixel count:', totalImageCount[0])

print()
print()

for row in range(0, 64):
	for col in range(0, 64):
		print(increasingCounts[64 * row + col], end=', ')
	print()

print()
print()
print()

for row in range(0, 64):
	for col in range(0, 64):
		print(decreasingCounts[64 * row + col], end=', ')
	print()