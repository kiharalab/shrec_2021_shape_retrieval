The code in this repository performs 3 task based on our teams contribution to the Shrec 2021 Surface-based protein domains retrieval track (http://shrec2021.drugdesign.fr/)

TASK 1: Generate the input 3DZD dataset
	We used 3DZD vectors to encode the feature for each samples in the training and test set.
	Code to generate input:
	python3 generate_input_3dzd.py path/to/queries_ply_shape/ path/to/queries_ply_electro/ path/to/output/generated/3dzd_features/
	The code takes 3 arguements :
		1. input ply shape file 
		2. input ply electrostatic file
		3. output directory to write 3DZD features for each sample

TASK 2: Train the neural network model for predicting (dis)-similarity between two pair of samples.
	Command :  python3 train.py --data_type 3dzd --iterations 500001 --evaluate_interval 100 --usebio true --traintype endtoend
	The code trains both the endtoend method and the extractor method as described in the report.
	Notable options to set when training are: 
	1. --usebio [true/false] --> this is the flag indicating if we want to use the biological information provided 
	2. --traintype [endtoend/extractor] --> this flag indicate the type of neural network model to train.
	More possible option can be found in the file header.

TASK 3: Generate prediction matrix for test set
	This step generate the (dis)-similarity matrix for all samples in the test set.
	Command: python3 generate_prediction.py --model_type neural_network --usebio true --traintype extractor --model_name extractor_shapeandbio
	The options for the code is similar to that of the training.
	The best model from the training should be copy to the Best_models directory.

A data directory is provided which contain the precomputed 3DZD input for test set of this track. 

question(s)/Inquires: contact Prof Daisuke Kihara at dkihara@purdue.edu
