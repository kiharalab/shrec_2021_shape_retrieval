#!/usr/bin/env python3

import sys
from itertools import product
import numpy as np
import numpy.linalg as npla
import scipy
from scipy.spatial import cKDTree

mesh_infilename = sys.argv[1]
values_infilename = sys.argv[2]
volume_n = int(sys.argv[3])
outfile_prefix = sys.argv[4]

def stream_ignore_blanks_comments(f):
	for line in f:
		x = line.find("#")
		if x != -1:
			line = line[:x]
		line = line.strip()
		if len(line) == 0:
			continue
		yield line

def read_meshdata_off(f):
	verts = []
	faces = []
	nverts = 0
	nfaces = 0
	nedges = 0
	if f.readline().strip() != "OFF":
		print("[error] OFF file header magic line missing")
		exit(1)
	for line in stream_ignore_blanks_comments(f):
		(nverts, nfaces, nedges) = (int(x) for x in line.split())
		break
	for i, line in zip(range(nverts), stream_ignore_blanks_comments(f)):
		verts.append(tuple(float(x) for x in line.split()[:3]))
	for i, line in zip(range(nfaces), stream_ignore_blanks_comments(f)):
		l = line.split()
		n = int(l[0])
		l = l[1:]
		faces.append(tuple(int(x) for x in l[:n]))
	return (verts, faces)

with open(mesh_infilename) as f:
	(verts, faces) = read_meshdata_off(f)

nverts = len(verts)
nfaces = len(faces)

with open(values_infilename) as f:
	vert_values = []
	for line in stream_ignore_blanks_comments(f):
		vert_values.append(tuple(float(x) for x in line.split()[:3]))

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

for ci in range(len(vert_values[0])):
	values = np.array([x[ci] for x in vert_values])
	def volume_interp_func_idx(xi, yi, zi):
		x = samples[xi,0]
		y = samples[yi,1]
		z = samples[zi,2]
		return volume_interp_func(x, y, z)
	def volume_interp_func(x, y, z):
		p = np.array((x,y,z))
		res = kdtree.query_ball_point(p, neighborhood_radius)
		if len(res) == 0:
			return 0.
		fullres = [(npverts[i], values[i]) for i in res]
		dists = [npla.norm(vert-p) for vert,val in fullres]
		dist2s = [d*d for d in dists]
		mini = np.argmin(dists)
		# if on top of known point, just use its value
		if npla.norm(npverts[mini]-p) < 1e-3:
			return values[mini]
		
		denom = sum(1/a for a in dist2s)
		funcval = sum(fr[1]/d2 for fr, d2 in zip(fullres, dist2s)) / denom
		return funcval
		
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
