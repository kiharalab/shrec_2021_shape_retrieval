import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import argparse
import numpy as np

from utils import read_json, evaluate, read_inv, read_3dzm,read_ply
from os.path import isfile, join
from models import SimpleEuclideanModel, NeuralNetworkModel
from models_attention import AttentionNeuralNetworkModel
from model_endtoend import EndToEndNeuralNetworkModel
from torch import FloatTensor, LongTensor

NB_TARGETS = 554#1543#3585
#BASE_OUTPUT_DIR = '/net/kihara-fast-scratch/taderinw/protein_shape/shrec2021/test_set_prediction/'
BASE_OUTPUT_DIR = '/net/kihara-fast-scratch/taderinw/protein_shape/original_track_2021/prediction/'

#BASE_FEATURE_DIR = '/net/kihara-fast-scratch/taderinw/protein_shape/shrec2021/OFF_training_anonym_v2/3dzm_data'
#inv_data = read_json(BASE_FEATURE_DIR + '/inv_info.json')
# ply_data = read_json('data/shrec_2019/ply_info.json')

# Argument Parsing
parser = argparse.ArgumentParser(description='Generating Predictions')
parser.add_argument('--model_type', type=str, default='simple_euclidean_model', help='Model Type')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--should_evaluate', type=str, default='true', help='Evaluate the model before inference')
parser.add_argument('--data_type',type=str, default='3dzd', help='The type of dataset to load. 3dzd or 3dzm')
parser.add_argument('--model_name', type=str, default='neural_network', help='Name of saved model to use for training')
parser.add_argument('--combine',type=str, default='false', help='This is use for 3dzm, if true, it will concatenate complex number components')
parser.add_argument('--usebio',type=str, default='false', help='This option is for combining biological data to shape for training')
parser.add_argument('--traintype',type=str, default='extractor', help='This is the option for using extractor or end to end')

def pairs_to_features_from_memory(pairs,ply,query, use_bio=False):
    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        id_0, id_1 = str(_pair[0]), str(_pair[1])
        _3dzd_1 = list(ply[id_0]['_3dzd'])
        _3dzd_2 = list(query[id_1]['_3dzd'])

        if use_bio:
            elec_1 = list(ply[id_0]['electrostatic'])
            # hydr_1 = list(dataset[id_0]['hydropathy'])
            # hbnd_1 = list(dataset[id_0]['hydrogen_bond'])

            elec_2 = list(query[id_1]['electrostatic'])
            # hydr_2 = list(dataset[id_1]['hydropathy'])
            # hbnd_2 = list(dataset[id_1]['hydrogen_bond'])


            _3dzd_1 = _3dzd_1 + elec_1 #+ hydr_1 + hbnd_1
            _3dzd_2 = _3dzd_2 + elec_2 #+ hydr_2 + hbnd_2


        _3DZD_vector_1 = np.asarray(_3dzd_1)
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = np.asarray(_3dzd_2)
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        element_vertex_1, element_face_1 = tuple([int(x) for x in ply[id_0]['vertex_face']])
        element_vertex_2, element_face_2 = tuple([int(x) for x in query[id_1]['vertex_face']])

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

# Main Function
def main():
    # Arguments Parsing
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true' and torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    device_id = args.device_id

    if args.should_evaluate == 'true':
        should_evaluate = True
    else:
        should_evaluate = False


    usebio = False
    if args.usebio == 'true':
        usebio = True

    model_type = args.model_type
    data_type = args.data_type
    combine = args.combine

    output_fn = BASE_OUTPUT_DIR +  '/' + model_type + '_' + args.model_name + '_' + data_type + '_bio_' + args.usebio + '_' + args.traintype + '_preds.matrix'
    print('Outputing to : ', output_fn)
    
    if combine == 'true':
        combine = True
    else:
        combine = False


    print('model_type = {}'.format(model_type))
    if model_type == 'neural_network':
        print('Neural Network Model')
        if isfile('Best_models/' + args.model_name):
            model = torch.load('Best_models/' + args.model_name)
            #print(model)
            if cuda: model.cuda(device_id)
            print('Best_models/' + args.model_name + ' Loaded and will be used for evaluation')
        else:
            print(args.model_name + ' Not found')
            exit()
    elif model_type == 'simple_euclidean_model':
        print('Simple Euclidean Model')
        model = SimpleEuclideanModel()
    model.eval()

    f = open(output_fn, 'w+')

    plydata = read_json('data/2021_original_track_ply.json')
    qdata = read_json('data/2021_original_track_query.json')
    for _id in plydata.keys():
            plydata[_id] = json.loads(plydata[_id])
    for _id in qdata.keys():
            qdata[_id] = json.loads(qdata[_id])

    for i in range(NB_TARGETS):
        print('Processing target {}'.format(i))
        my_pairs = [(i, j) for j in range(1,11)]
        
        if data_type == '3dzd':
            # inputs_1, inputs_2, extra_features = pairs_to_features(my_pairs,ply_data)
            inputs_1, inputs_2, extra_features = pairs_to_features_from_memory(my_pairs,plydata,qdata,usebio)
        else:
            pass
            # inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(my_pairs,ply_data,_3dzm_data,combine)
            #inputs_1, inputs_2, extra_features = pairs_to_features_from_memory(my_pairs,fulldata,usebio)

        if cuda:
            inputs_1 = inputs_1.cuda(device_id)
            inputs_2 = inputs_2.cuda(device_id)
            extra_features = extra_features.cuda(device_id)

        outputs = model(inputs_1, inputs_2, extra_features, True)
        outputs = outputs.squeeze().cpu().data.numpy().tolist()
        outputs = [str(output) for output in outputs]
        output_str =  ' '.join(outputs)
        f.write(output_str)
        f.write('\n')
    f.close()

if __name__=="__main__":
    main()
