#!/usr/bin/env python3

import sys
import os
from itertools import product
import numpy as np
import numpy.linalg as npla
import scipy
from scipy.spatial import cKDTree
import gzip
import pandas as pd


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


def convert_csv_and_values_to_situs(mesh_infilename,volume_n,outfile_prefix,opath):
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
            (discard, res) = kdtree.query(p, k=5, distance_upper_bound=neighborhood_radius)
            res = [o for o in res if o != kdtree.n]
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
        
        ofname = opath + "%s_column%d_interp_shell.situs" % (outfile_prefix, ci)
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


def convert_csv_and_values_to_situs_shapeonly(mesh_infilename,volume_n,outfile_prefix,opath):
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
        
        ofname = opath + "%s_column%d_interp_shell.situs" % (outfile_prefix, ci)
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




def convert_to_csv(inf,outp,offset,type=0):
    _id = inf.split('/')[-1]
    lines = open(inf).readlines()
    lines = [list(float(b) for b in x.strip().split(',')) for x in lines[offset::]]
    df = pd.DataFrame(lines)
    df = df.dropna(axis=1)
    if type == 1:
        df[4] = [1] * df.shape[0]
    #df.to_csv(outp + _id + '.csv',sep=',',header=False,index=False)
    df.to_csv(inf,sep=',',header=False,index=False)

    
# Actual generation begins
off = sys.argv[1]
bio = sys.argv[2]
outpath = sys.argv[3]
ltp = sys.argv[4]
mylist = ltp
mylist = [x.strip() for x in open(mylist).readlines()]
mylist = [x.split('.')[0] for  x in mylist]

offlist = [y for y in os.listdir(off) if '.csv' in y]

for item in offlist:
    if item.split('.')[0] in mylist:
        
        struct_id = item.split('.')[0]
        print('processing : ', struct_id)

        mesh_infilename = off + '/' + item
        values_infilename = bio+ '/' + item #+ '.electro.ply'
        volume_n = 256
        outfile_prefix = struct_id + '_dim' + str(volume_n)
        shapeoutfile_prefix = struct_id + '_shape_dim' + str(volume_n)
        #print(mesh_infilename)
        #convert_to_csv(mesh_infilename,outpath,0,1)
        
        #convert_to_csv(values_infilename,outpath,11)

        shape_csv  = mesh_infilename 
        values_csv = values_infilename.replace('shape','electro')
        print(shape_csv,values_csv)
        convert_csv_and_values_to_situs(values_csv,volume_n,outfile_prefix,outpath)
        convert_csv_and_values_to_situs_shapeonly(shape_csv,volume_n,shapeoutfile_prefix,outpath)
    
        cmd = 'gzip '+outpath + struct_id+'_dim256_column0_interp_shell.situs'
        os.system(cmd)
        cmd = 'gzip '+outpath + struct_id+'_shape_dim256_column0_interp_shell.situs'
        os.system(cmd)
        os.system('nohup python3 bin/grid2zerd.py ' + ltp + ' > ' + ltp + '.lzerd.out &')
