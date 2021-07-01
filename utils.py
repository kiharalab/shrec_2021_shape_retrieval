import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import numpy as np

from os.path import isfile, join
from torch import FloatTensor, LongTensor

BASE_FEATURE_DIR = 'path/to/3dzd_data'

def matrix_to_ranking(matrix_fp):
    ranking_fp = matrix_fp.replace('.matrix', '.ranking')
    f = open(ranking_fp, 'w+')
    matrix = np.loadtxt(matrix_fp)
    nb_targets = matrix.shape[0]
    for i in range(nb_targets):
        dists = matrix[i,:].tolist()
        indexes = np.argsort(dists)
        indexes = [str(index) for index in indexes]
        indexes_str = ' '.join(indexes)
        f.write('{}\n'.format(indexes_str))
    f.close()

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_map(scores, correct_labels):
    # Sorting
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            if scores[i] < scores[j]:
                scores[i], scores[j] = scores[j], scores[i]
                correct_labels[i], correct_labels[j] = correct_labels[j], correct_labels[i]
    # Calculate average_precision
    nb_relevants = sum(correct_labels)
    average_precision = 0
    relevant_count = 0
    for i in range(len(correct_labels)):
        if correct_labels[i] == 1:
            relevant_count += 1
            average_precision += (relevant_count / float(i+1))
    average_precision /= float(nb_relevants)

    return average_precision

def evaluate(model, test_fp, data_type, cuda=None, device_id=None, combine=False):
    query2candidates = {}
    with open(test_fp) as f:
        for line in f:
            query, candidate, label = line.strip().split()
            if not query in query2candidates:
                query2candidates[query] = {}
                query2candidates[query]['positives'] = []
                query2candidates[query]['negatives'] = []
            if int(label) == 0: query2candidates[query]['negatives'].append(candidate)
            if int(label) == 1: query2candidates[query]['positives'].append(candidate)

    average_precisions = []
    queries = list(query2candidates.keys())
    for query in queries:
        negative_candidates = query2candidates[query]['negatives']
        positive_candidates = query2candidates[query]['positives']
        candidates = negative_candidates + positive_candidates
        correct_labels = [0] * len(negative_candidates) + [1] * len(positive_candidates)
        pairs = [(query, candidate) for candidate in candidates]
        
        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(pairs,combine)
        
        #inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        if cuda:
            inputs_1 = inputs_1.cuda(device_id)
            inputs_2 = inputs_2.cuda(device_id)
            extra_features = extra_features.cuda(device_id)

        scores = model(inputs_1, inputs_2, extra_features)
        scores = scores.squeeze().cpu().data.numpy().tolist()
        # Sorting
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                if scores[i] < scores[j]:
                    scores[i], scores[j] = scores[j], scores[i]
                    correct_labels[i], correct_labels[j] = correct_labels[j], correct_labels[i]
        # Calculate average_precision
        nb_relevants = len(positive_candidates)
        average_precision = 0
        relevant_count = 0
        for i in range(len(candidates)):
            if correct_labels[i] == 1:
                relevant_count += 1
                average_precision += (relevant_count / float(i+1))
        average_precision /= float(nb_relevants)

        # Update
        average_precisions.append(average_precision)
    map_score = np.average(average_precisions)
    return map_score
    
    
def evaluate_v2(model, criterion, test_fp, data_type, cuda=None, device_id=None, combine=False):
    query2candidates = {}
    with open(test_fp) as f:
        for line in f:
            query, candidate, label = line.strip().split()
            if not query in query2candidates:
                query2candidates[query] = {}
                query2candidates[query]['positives'] = []
                query2candidates[query]['negatives'] = []
            if int(label) == 0: query2candidates[query]['negatives'].append(candidate)
            if int(label) == 1: query2candidates[query]['positives'].append(candidate)

    average_precisions = []
    average_loss = []
    queries = list(query2candidates.keys())
    for query in queries:
        negative_candidates = query2candidates[query]['negatives']
        positive_candidates = query2candidates[query]['positives']
        candidates = negative_candidates + positive_candidates
        correct_labels = [0] * len(negative_candidates) + [1] * len(positive_candidates)
        pairs = [(query, candidate) for candidate in candidates]
        
        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(pairs,combine)
        
        #inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        if cuda:
            inputs_1 = inputs_1.cuda(device_id)
            inputs_2 = inputs_2.cuda(device_id)
            extra_features = extra_features.cuda(device_id)

        scores = model(inputs_1, inputs_2, extra_features)
        score_raw = scores
        scores = scores.squeeze().cpu().data.numpy().tolist()
        # Sorting
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                if scores[i] < scores[j]:
                    scores[i], scores[j] = scores[j], scores[i]
                    correct_labels[i], correct_labels[j] = correct_labels[j], correct_labels[i]
        
        #Calculating loss
        label_tensor = FloatTensor(correct_labels).unsqueeze(1)
        label_tensor = label_tensor.cuda(device_id)
        loss = criterion(score_raw, label_tensor)
        average_loss.append(np.float(loss.squeeze().cpu().data))
        
        
        # Calculate average_precision
        nb_relevants = len(positive_candidates)
        average_precision = 0
        relevant_count = 0
        for i in range(len(candidates)):
            if correct_labels[i] == 1:
                relevant_count += 1
                average_precision += (relevant_count / float(i+1))
        average_precision /= float(nb_relevants)
        
        # Update
        average_precisions.append(average_precision)
    map_score = np.average(average_precisions)
    avg_loss = np.average(average_loss)
    return map_score, avg_loss

def evaluate_resnet(model, test_fp, data_type, cuda=None, device_id=None, combine=False):
    query2candidates = {}
    with open(test_fp) as f:
        for line in f:
            query, candidate, label = line.strip().split()
            if not query in query2candidates:
                query2candidates[query] = {}
                query2candidates[query]['positives'] = []
                query2candidates[query]['negatives'] = []
            if int(label) == 0: query2candidates[query]['negatives'].append(candidate)
            if int(label) == 1: query2candidates[query]['positives'].append(candidate)

    average_precisions = []
    queries = list(query2candidates.keys())
    for query in queries:
        negative_candidates = query2candidates[query]['negatives']
        positive_candidates = query2candidates[query]['positives']
        candidates = negative_candidates + positive_candidates
        correct_labels = [0] * len(negative_candidates) + [1] * len(positive_candidates)
        pairs = [(query, candidate) for candidate in candidates]
        
        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        else:
            inputs_1, inputs_2, extra_features = read_3dzm_for_resnet_mem(pairs)
        
        #inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        if cuda:
            inputs_1 = inputs_1.cuda(device_id)
            inputs_2 = inputs_2.cuda(device_id)
            extra_features = extra_features.cuda(device_id)

        scores = model(inputs_1, inputs_2, extra_features)
        scores = scores.squeeze().cpu().data.numpy().tolist()
        # Sorting
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                if scores[i] < scores[j]:
                    scores[i], scores[j] = scores[j], scores[i]
                    correct_labels[i], correct_labels[j] = correct_labels[j], correct_labels[i]
        # Calculate average_precision
        nb_relevants = len(positive_candidates)
        average_precision = 0
        relevant_count = 0
        for i in range(len(candidates)):
            if correct_labels[i] == 1:
                relevant_count += 1
                average_precision += (relevant_count / float(i+1))
        average_precision /= float(nb_relevants)

        # Update
        average_precisions.append(average_precision)
    map_score = np.average(average_precisions)
    return map_score

def pairs_to_features(pairs):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        #check_file = read_inv_file(join(BASE_FEATURE_DIR, '{}.inv'.format(_pair[0])))
        #if not check_file:
        #    continue
        _3DZD_vector_1 = read_inv(join(BASE_FEATURE_DIR, '{}.inv'.format(_pair[0])))
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = read_inv(join(BASE_FEATURE_DIR, '{}.inv'.format(_pair[1])))
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        element_vertex_1, element_face_1 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[0])))
        element_vertex_2, element_face_2 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[1])))

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1).squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2).squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features

def pairs_to_features_from_memory(pairs,dataset, use_bio=False):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        id_0, id_1 = str(_pair[0]), str(_pair[1])
        _3dzd_1 = list(dataset[id_0]['_3dzd'])
        _3dzd_2 = list(dataset[id_1]['_3dzd'])

        if use_bio:
            elec_1 = list(dataset[id_0]['electrostatic'])
            # hydr_1 = list(dataset[id_0]['hydropathy'])
            # hbnd_1 = list(dataset[id_0]['hydrogen_bond'])

            elec_2 = list(dataset[id_1]['electrostatic'])
            # hydr_2 = list(dataset[id_1]['hydropathy'])
            # hbnd_2 = list(dataset[id_1]['hydrogen_bond'])


            _3dzd_1 = _3dzd_1 + elec_1 #+ hydr_1 + hbnd_1
            _3dzd_2 = _3dzd_2 + elec_2 #+ hydr_2 + hbnd_2


        _3DZD_vector_1 = np.asarray(_3dzd_1)
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = np.asarray(_3dzd_2)
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        element_vertex_1, element_face_1 = tuple([int(x) for x in dataset[id_0]['vertex_face']])
        element_vertex_2, element_face_2 = tuple([int(x) for x in dataset[id_1]['vertex_face']])

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1).squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2).squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features

def pairs_to_features_3dzm(pairs, combine=False):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        #check_file = read_inv_file(join(BASE_FEATURE_DIR, '{}.inv'.format(_pair[0])))
        #if not check_file:
        #    continue
        _3DZD_vector_1 = read_3dzm(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[0])),combine)
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = read_3dzm(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[1])),combine)
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        element_vertex_1, element_face_1 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[0])))
        element_vertex_2, element_face_2 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[1])))

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1).squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2).squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features



def read_3dzm_for_resnet(pairs,_3dzm,_ply,combine=False):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        id_0, id_1 = str(_pair[0]), str(_pair[1])
        _3DZD_vector_1 = np.asarray(_3dzm[id_0])#read_3dzm_resnet(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[0])))
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = np.asarray(_3dzm[id_1])#read_3dzm_resnet(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[1])))
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)


        # element_vertex_1, element_face_1 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[0])))
        # element_vertex_2, element_face_2 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[1])))

        # element_vertex_1, element_face_1 = _ply[id_0]['element_vertex'], _ply[id_0]['element_face']
        # element_vertex_2, element_face_2 = _ply[id_1]['element_vertex'], _ply[id_1]['element_face']
        element_vertex_1, element_face_1 = _ply[id_0][0], _ply[id_0][1]
        element_vertex_2, element_face_2 = _ply[id_1][0], _ply[id_1][1]

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1)#.squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2)#.squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features

def read_3dzm_for_resnet_mem(pairs,combine=False):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        _3DZD_vector_1 = read_3dzm_resnet(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[0])))
        #_3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = read_3dzm_resnet(join(BASE_FEATURE_DIR, '{}.3dzm'.format(_pair[1])))
        #_3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)


        element_vertex_1, element_face_1 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[0])))
        element_vertex_2, element_face_2 = read_ply(join(BASE_FEATURE_DIR, '{}.ply'.format(_pair[1])))

        # element_vertex_1, element_face_1 = _ply[id_0]['element_vertex'], _ply[id_0]['element_face']
        # element_vertex_2, element_face_2 = _ply[id_1]['element_vertex'], _ply[id_1]['element_face']
        # element_vertex_1, element_face_1 = _ply[id_0][0], _ply[id_0][1]
        # element_vertex_2, element_face_2 = _ply[id_1][0], _ply[id_1][1]

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1)#.squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2)#.squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features

def read_lines(fn):
    lines = []
    f = open(fn, 'r')
    for line in f: lines.append(line.strip())
    f.close()
    return lines

def read_org_test_list():
    lines = []
    f = open('data/2021_test_list')
    for l in f.readlines():
        lines.append(l.strip('\n'))
    f.close()
    return lines

def read_test_set_pair():
    pairs = []
    with open('data/2021_test_pair_list') as f:
        for item in f.readlines():
            pairs.append(item.split())
    return pairs

def read_test_set_pair_scope():
    pairs = []
    with open('data/text_v2.txt') as f:
        for item in f.readlines():
            pairs.append(item.split())
    return pairs
def read_exclusion():
    lines = []
    #f = open('failed_list')
    f2 = open('artifact_class')
    #for l in f.readlines():
    #    lines.append(l.strip('\n'))
    for l in f2.readlines():
        lines.append(l.strip('\n'))
    #f.close()
    f2.close()
    return lines

def read_json(fn):
    with open(fn) as f:
        return json.load(f)

def read_off(fn):
    f = open(fn, 'r')
    f.readline()
    elements = f.readline().strip().split()
    element_vertex, element_face = int(elements[0]), int(elements[1])
    f.close()
    return (element_vertex, element_face)

def read_ply(fn):
    element_vertex = None
    element_face = None
    f = open(fn, 'r')
    for line in f:
        line = line.strip()
        if 'element vertex' in line:
            element_vertex = int(line.split()[-1])
        if 'element face' in line:
            element_face = int(line.split()[-1])
        if element_vertex != None and element_face != None:
            return (element_vertex, element_face)
    f.close()
    return (element_vertex, element_face)

def read_inv(fn):
    vectors = []
    first_line = True
    f = open(fn, 'r')
    for line in f:
        line = line.strip()
        if len(line) == 0: continue
        if first_line:
            first_line = False
        else:
            vectors.append(np.float(line.strip()))
    f.close()
    return np.asarray(vectors)

def read_3dzm(fn,combine):
    dzm = []
    f = open(fn,'r')
    for lines in f.readlines():
        data = lines.split(' ')[-1].strip('(').strip('\n')
        data = data.replace(')','')
        data = data.split(',')
        data = [float(x) for x in data]
        if combine:
            dzm.append(data[0])
            dzm.append(data[1])
        else:
            dzm.append(abs(complex(data[0],data[1])))

    return np.asarray(dzm)

def read_3dzm_resnet(fn):
    dzm = []
    d1 = []
    d2 = []
    f = open(fn,'r')
    for lines in f.readlines():
        data = lines.split(' ')[-1].strip('(').strip('\n')
        data = data.replace(')','')
        data = data.split(',')
        data = [float(x) for x in data]
        d1.append(data[0])
        d2.append(data[1])

    dzm.append(d1)
    dzm.append(d2)

    dzm = np.asarray(dzm)
    dzm = np.expand_dims(dzm.squeeze(), axis = 0)


    return dzm



def read_inv_file(fn):
    try:
        if os.path.isfile(fn):
            return True
        else:
            return False
    except Exception as e:
        return False


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        return len(self.items)
