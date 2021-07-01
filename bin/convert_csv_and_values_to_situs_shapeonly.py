#!/usr/bin/env python3

import sys
from itertools import product
import numpy as np
import numpy.linalg as npla
import scipy
from scipy.spatial import cKDTree

mesh_infilename = sys.argv[1]
#values_infilename = sys.argv[2]
volume_n = int(sys.argv[2])
outfile_prefix = sys.argv[3]

def stream_ignore_blanks_comments(f):
	for line in f:
		x = line.find("#")
		if x != -1:
			line = line[:x]
		line = line.strip()
		if len(line) == 0:
			continue
		yield line

def read_meshdata_csv(f):
	verts = []
	faces = []
	nverts = 0
	nfaces = 0
	nedges = 0
	values = []
	for line in stream_ignore_blanks_comments(f):
		l = line.split(",")
		verts.append(tuple(float(x) for x in l[:3]))
		values.append(float(l[3]))
	return (verts, faces, values)

with open(mesh_infilename) as f:
	(verts, faces, values) = read_meshdata_csv(f)

nverts = len(verts)
nfaces = len(faces)

vert_values = [(x,) for x in values]

# all data is loaded, begin processing
npverts = [np.array(x) for x in verts]
kdtree = cKDTree(npverts)

center = np.zeros(3)
for x in npverts:
	center += x
center = center / nverts

tight_radius = max(npla.norm(x-center) for x in npverts)
sampling_radius = 1.1 * tight_radius # 10% padding fudge factor
voxel_spacing = (2*sampling_radius) / volume_n

samples = np.linspace(center-sampling_radius, center+sampling_radius, volume_n)
#print(samples)

# use cutoff from 3D-Surfer
neighborhood_radius = 1.7 * voxel_spacing

#volume = np.zeros((volume_n, volume_n, volume_n))
#volume = np.fromfunction()

for ci in range(len(vert_values[0])):
	values = np.array([x[ci] for x in vert_values])
	def volume_interp_func_idx(xi, yi, zi):
		x = samples[xi,0]
		y = samples[yi,1]
		z = samples[zi,2]
		return volume_interp_func(x, y, z)
	def volume_interp_func(x, y, z):
		p = np.array((x,y,z))
		(discard, res) = kdtree.query(p, k=2, distance_upper_bound=neighborhood_radius)
		res = [o for o in res if o != kdtree.n]
		if len(res) == 0:
			return 0.
		else:
			return 1.
		
	volume = np.zeros((volume_n, volume_n, volume_n))
	for i,j,k in product(range(volume_n), range(volume_n), range(volume_n)):
		volume[i,j,k] = volume_interp_func_idx(i,j,k)
	
	ofname = "%s_column%d_interp_shell.situs" % (outfile_prefix, ci)
	with open(ofname, "w") as f:
		print(voxel_spacing, 
			samples[0,0], samples[0,1], samples[0,2], volume_n, volume_n, volume_n, 
			file=f)
		count = 0
		for vvv in volume.reshape((volume_n*volume_n*volume_n,)):
			count += 1
			if count % 10 == 0:
				print(vvv, file=f)
			else:
				f.write(str(vvv)+" ")
	print("[info] wrote output file '%s'" % (ofname))
