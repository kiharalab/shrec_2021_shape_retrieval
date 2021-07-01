import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import sys
from os.path import isfile, join

from models import NeuralNetworkModel
from models_attention import AttentionNeuralNetworkModel
from model_endtoend import EndToEndNeuralNetworkModel
from reader import Dataset
from reader_organizer import OrganizerDataset
from utils import evaluate,evaluate_v2,calculate_map, create_dir_if_not_exists

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--iterations', type=int, default=10000, help='Number training iterations')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='Learning rate for optimizers')
parser.add_argument('--log_interval', type=int, default=50, help='Print status every log_interval iterations.')
parser.add_argument('--evaluate_interval', type=int, default=50, help='Evaluate every evaluate_interval iterations.')
parser.add_argument('--resume', type=int, default=0, help='Resume training with a saved model or not.')
parser.add_argument('--model_name', type=str, default='neural_network', help='Name of saved model to use for training')
parser.add_argument('--data_type',type=str, default='3dzd', help='The type of dataset to load. 3dzd or 3dzm')
parser.add_argument('--attention', type=str, default='false', help='Use Attention on the encoder outputs')
parser.add_argument('--combine',type=str, default='false', help='This is use for 3dzm, if true, it will concatenate complex number components')
parser.add_argument('--usebio',type=str, default='false', help='This option is for combining biological data to shape for training')
parser.add_argument('--traintype',type=str, default='extractor', help='This is the option for using extractor or end to end')

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

    iterations = args.iterations
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_interval = args.log_interval
    evaluate_interval = args.evaluate_interval
    data_type = args.data_type
    combine = args.combine

    use_attention = str(args.attention)

    data_size_multiplier = 1

    # Load dataset
    dataset = Dataset()
    print('Loaded dataset')
    
    if args.usebio == 'true':
        dataset.set_usebio(True)
        data_size_multiplier = 3

    if combine == 'true':
        combine = True
    else:
        combine = False

    if data_type == '3dzd':
        input_size = 121 * data_size_multiplier
    else:
        if combine:
            input_size = 2 * 1771 * data_size_multiplier
        else:
            input_size = 1771 * data_size_multiplier

    # Load model
    # The input 3DZD calculation consists of 121 numbers
    # Extra features consists of 2 numbers (vertices_diff and faces_diff)
    if args.resume and isfile('models/' + args.model_name):
        model = torch.load('models/' + args.model_name)
        print('Resuming training with saved model : ' + args.model_name)
    else:
        print('Starting training with fresh model')
        if args.attention == 'true':
            model = AttentionNeuralNetworkModel(input_dim=input_size, hidden_dims=[250, 200, 150],fc_dims=[100, 50], extra_feature_dim=2)
            print('Using attention model for training')
        else:
            if args.traintype == 'extractor':
                model = NeuralNetworkModel(input_dim=input_size, hidden_dims=[250, 200, 150],fc_dims=[100, 50], extra_feature_dim=2)
                print('using extractor model')
            else:
                model = EndToEndNeuralNetworkModel(input_dim=input_size, hidden_dims=[250, 200, 150],fc_dims=[100, 50], extra_feature_dim=2)
                print('using end to end model')


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    if cuda:
        model.cuda(device_id)
    print('Loaded model')

    # Start Training
    losses = []
    map_avg = []
    best_map_score = 0.0

    # Set up logs and saving directories
    create_dir_if_not_exists('training_log/')
    create_dir_if_not_exists('models/')
    
    for itx in range(iterations):
        sys.stdout.flush()
        print('Training itx = {}'.format(itx))
        model.train()
        model.zero_grad()

        nb_positives = nb_negatives = batch_size // 2
        inputs_1, inputs_2, extra_features, labels = \
            dataset.next_batch(data_type,nb_positives, nb_negatives,combine)

        if cuda:
            inputs_1 = inputs_1.cuda(device_id)
            inputs_2 = inputs_2.cuda(device_id)
            extra_features = extra_features.cuda(device_id)
            labels = labels.cuda(device_id)

        outputs = model(inputs_1, inputs_2, extra_features)
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        losses.append(float(loss.squeeze().cpu().data))
        
        # Calculating MAP for Training 
        train_scores = outputs.squeeze().cpu().data.numpy().tolist()
        train_label = labels.squeeze().cpu().data.numpy().tolist()
        average_precision = calculate_map(train_scores,train_label)
        map_avg.append(average_precision)
        print("---------------------")
        print("iters:", itx)
        print("Train Average loss:", np.average(losses))
        print("Train Batch Average Map score :", average_precision)
        print("Train Average Map Score :", np.average(map_avg))
        
        if itx % evaluate_interval == 0:
            model.eval()

            if args.attention == 'false':
                test_inputs_1, test_inputs_2, test_extra_features, test_labels = dataset.test_set_batch(data_type)
            else:
                test_inputs_1, test_inputs_2, test_extra_features, test_labels = dataset.test_set_batch_attention(data_type)

            if cuda:
                test_inputs_1 = test_inputs_1.cuda(device_id)
                test_inputs_2 = test_inputs_2.cuda(device_id)
                test_extra_features = test_extra_features.cuda(device_id)
                test_labels = test_labels.cuda(device_id)

            test_outputs = model(test_inputs_1, test_inputs_2, test_extra_features)
            test_loss = criterion(test_outputs, test_labels)
            test_loss = float(test_loss.squeeze().cpu().data)

            test_labels = test_labels.squeeze().cpu().data.numpy().tolist()
            test_scores = test_outputs.squeeze().cpu().data.numpy().tolist()
            map_score = calculate_map(test_scores,test_labels)
            
            print("Test Loss on test set:", test_loss)
            print("Map Score on test set:", map_score)
            
            log_str = "iters:\t" + str(itx) + \
             "\tTrain Average loss:" + str(np.average(losses)) + \
             "\tTraining Average Map Score:" +str(np.average(map_avg)) +  \
             "\tTraining Batch Average Map Score:" +str(average_precision) +  \
             "\tMap Score on test set:" + str(map_score)+ \
             "\tTest Loss on test set:" + str(test_loss) \
             + "\n"
                
            with open('training_log/Original_good_3dzd_data_' + str(batch_size) + '_' + data_type + '_' + args.traintype + '_usebio_' +str(args.usebio) +'_longer_attention_'+use_attention+'.log','a+') as f:
                f.write(log_str)
                
            if map_score > best_map_score:
                best_map_score = map_score
                torch.save(model, 'models/Original_good_3dzd_data_'+ str(batch_size) + '_' +data_type + '_' + args.traintype + '_' + str(itx) + '_usebio_' + str(args.usebio) + '_longer_attention_'+use_attention)
                print("Save the model")
                sys.stdout.flush()

            if args.attention == 'true' and itx % 5000 == 0:
                torch.save(model, 'models/Original_good_3dzd_data_'+ str(batch_size) + '_' +data_type + '_' + args.traintype + '_' + str(itx) + '_usebio_' + str(args.usebio) + '_longer_attention_'+use_attention)
                print("Save the model")
                sys.stdout.flush()

        loss.backward()
        optimizer.step()
            

if __name__=="__main__":
    main()
