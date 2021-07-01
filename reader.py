import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import numpy as np

from os.path import isfile, join
from utils import read_json, read_inv, read_ply, pairs_to_features, pairs_to_features_3dzm, read_3dzm_for_resnet, read_exclusion, AugmentedList,pairs_to_features_from_memory,read_test_set_pair_scope,read_org_test_list

from torch import FloatTensor, LongTensor

# Dataset Class
class Dataset:
    def __init__(self):
        protein2ids, species2ids = {}, {}
        data = read_json('data/scope_data_norm_correct.json')
        print('data json loaded')
        excluded_list = read_exclusion()
        test_set_list =[x.split() for x in open('data/scope_test_list.txt').readlines()]
        missing_test = [x.split() for x in open('data/missing_test_list').readlines()]
        test_set_list = [x for x in test_set_list if x not in missing_test]
        self.all_ids = []
        count = 0
        for _id in data.keys():
            if _id in excluded_list:
                continue
            # if count == 100:
            #     break
            count += 1
            data[_id] = json.loads(data[_id])
            if _id in test_set_list:
                continue
            self.all_ids.append(_id)
            protein = data[_id]['protein']
            species = data[_id]['species']
            if not protein in protein2ids: protein2ids[protein] = []
            if not species in species2ids: species2ids[species] = []
            protein2ids[protein].append(_id)
            species2ids[species].append(_id)
        print(len(self.all_ids))
        self.data = data
        #self.all_ids = [x for x in list(data.keys()) if x not in excluded_list]
        self.protein2ids = protein2ids
        self.species2ids = species2ids

        self.testset = test_set_list
        self.usebio = False
        self.test_pairs = read_test_set_pair_scope()
        longlist = []
        for i in self.test_pairs:
            longlist.append(i[0])
            longlist.append(i[1])


        positive_pairs = []
        negative_pairs = []
        for item in longlist:
            if isinstance(self.data[item], str):
                self.data[item] = json.loads(self.data[item])
        for item in self.test_pairs:
            if item[0] in missing_test or item[1] in missing_test:
                continue
            
            if int(item[2]) == 1:
                positive_pairs.append((item[0],item[1]))
            else:
                negative_pairs.append((item[0],item[1]))
        
        positive_labels = [1] * len(positive_pairs)
        negative_labels = [0] * len(negative_pairs)
        positive_examples = list(zip(positive_pairs, positive_labels))
        negative_examples = list(zip(negative_pairs, negative_labels))

        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)

        pairs = [example[0] for example in all_examples]
        labels = [example[1] for example in all_examples]
        self.test_pairs = pairs
        self.test_pairs_label = labels


    def set_usebio(self,value):
        self.usebio = value
        print('Use bio data status', self.usebio)

    def sample_positive_pairs(self, nb_postives):
        positive_pairs = []
        for _ in range(nb_postives):
            selected_id = random.choice(self.all_ids)
            _protein_ids = set(self.protein2ids[self.data[selected_id]['protein']])
            _species_ids = set(self.species2ids[self.data[selected_id]['species']])
            candidates = list(_protein_ids.union(_species_ids))
            other_id = random.choice(candidates)
            positive_pairs.append((selected_id, other_id))
        return positive_pairs

    def sample_negative_pairs(self, nb_negatives):
        data = self.data
        negative_pairs = []
        for _ in range(nb_negatives):
            while True:
                samples = random.sample(self.all_ids, 2)
                if data[samples[0]]['protein'] != data[samples[1]]['protein'] and \
                data[samples[0]]['species'] != data[samples[1]]['species']:
                    negative_pairs.append(samples)
                    break
        return negative_pairs

    def next_batch(self, data_type, nb_postives=64, nb_negatives=64,combine=False):
        assert(nb_postives > 0)
        assert(nb_negatives > 0)

        positive_pairs = self.sample_positive_pairs(nb_postives)
        positive_labels = [1] * nb_postives
        positive_examples = list(zip(positive_pairs, positive_labels))

        negative_pairs = self.sample_negative_pairs(nb_negatives)
        negative_labels = [0] * nb_negatives
        negative_examples = list(zip(negative_pairs, negative_labels))

        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)

        pairs = [example[0] for example in all_examples]
        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features_from_memory(pairs,self.data,self.usebio)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(pairs,combine)
        
        # This will never happen again... just leaving it in here...
        if inputs_1.size()[0] < nb_postives or inputs_2.size()[0] < nb_negatives:
            print('Number of sampled pairs != Number of labels... resampling')
            next_batch(data_type,nb_postives,nb_negatives)
        

        labels = [example[1] for example in all_examples]
        labels = FloatTensor(labels).unsqueeze(1)

        return inputs_1, inputs_2, extra_features, labels

    def next_batch_old(self, data_type, nb_postives=64, nb_negatives=64,combine=False):
        assert(nb_postives > 0)
        assert(nb_negatives > 0)

        positive_pairs = self.sample_positive_pairs(nb_postives)
        positive_labels = [1] * nb_postives
        positive_examples = list(zip(positive_pairs, positive_labels))

        negative_pairs = self.sample_negative_pairs(nb_negatives)
        negative_labels = [0] * nb_negatives
        negative_examples = list(zip(negative_pairs, negative_labels))

        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)

        pairs = [example[0] for example in all_examples]
        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features(pairs)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(pairs,combine)
        
        # This will never happen again... just leaving it in here...
        if inputs_1.size()[0] < nb_postives or inputs_2.size()[0] < nb_negatives:
            print('Number of sampled pairs != Number of labels... resampling')
            next_batch(data_type,nb_postives,nb_negatives)
        

        labels = [example[1] for example in all_examples]
        labels = FloatTensor(labels).unsqueeze(1)

        return inputs_1, inputs_2, extra_features, labels

    def test_set_batch(self, data_type):
        pairs = self.test_pairs
        labels = self.test_pairs_label

        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features_from_memory(pairs,self.data,self.usebio)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(pairs,combine)
        
        labels = FloatTensor(labels).unsqueeze(1)

        return inputs_1, inputs_2, extra_features, labels

    def test_set_batch_attention(self, data_type):
        pairs = self.test_pairs
        labels = self.test_pairs_label
        test_samples = random.sample(pairs, 64)
        test_labels = []
        for item in test_samples:
            l = labels[pairs.index(item)]
            test_labels.append(l)

        if data_type == '3dzd':
            inputs_1, inputs_2, extra_features = pairs_to_features_from_memory(test_samples,self.data,self.usebio)
        else:
            inputs_1, inputs_2, extra_features = pairs_to_features_3dzm(test_samples,combine)
        
        labels = FloatTensor(test_labels).unsqueeze(1)

        return inputs_1, inputs_2, extra_features, labels


    def next_batch_resnet(self, data_type, nb_postives=64, nb_negatives=64,combine=False):
        assert(nb_postives > 0)
        assert(nb_negatives > 0)

        positive_pairs = self.sample_positive_pairs(nb_postives)
        positive_labels = [1] * nb_postives
        positive_examples = list(zip(positive_pairs, positive_labels))

        negative_pairs = self.sample_negative_pairs(nb_negatives)
        negative_labels = [0] * nb_negatives
        negative_examples = list(zip(negative_pairs, negative_labels))

        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)

        pairs = [example[0] for example in all_examples]
        
        inputs_1, inputs_2, extra_features = read_3dzm_for_resnet(pairs,self._3dzm_data,self.ply_data,combine)      

        labels = [example[1] for example in all_examples]
        labels = FloatTensor(labels).unsqueeze(1)

        return inputs_1, inputs_2, extra_features, labels


    def load_3dzm_to_memory(self):
        self._3dzm_data = read_json('data/scope_3dzm_data.json')


    def load_ply_to_memory(self):
        self.ply_data = read_json('data/scope_ply_data.json')

    def load_3dzd_to_meory(self):
        self._3dzd_data = read_json('data/scope_3dzd_data.json')
