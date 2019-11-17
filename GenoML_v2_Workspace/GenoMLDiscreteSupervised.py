# Importing the necessary packages 
import argparse
import sys
import sklearn
import h5py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

# Importing the necessary GenoML-specific packages 
from genoml.discrete.supervised import train
from genoml.discrete.supervised import tune

# Parsing arguments inputted by user 
# TRAINING
parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--prefix', type=str, default='GenoML_data', help='Prefix you would like to define for your training data build.')
parser.add_argument('--rank-features', type=str, default='skip', help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow with huge numbers of features [default: skip].')

# TUNING 
parser.add_argument('--max-tune', type=int, default=50, help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
parser.add_argument('--n-cv', type=int, default=5, help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')


# Actual parsing of arguments 
args = parser.parse_args()

# TRAINING
# Create a dialogue with the user 
print("")
print("Here is some basic info on the command you are about to run.")
print("Python Version info...")
print(sys.version) #TODO: Add switch here to only run in Python 3.x 

print("CLI argument info...")
print("Are you ranking features, even though it is pretty slow? Right now, GenoML runs general recursive feature ranking. You chose to", args.rank_features, "this part.")
print("Working with dataset", args.prefix, "from previous data munging efforts.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to Python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")
print("")

run_prefix = args.prefix
infile_h5 = run_prefix + ".dataForML.h5"
df = pd.read_hdf(infile_h5, key = "dataForML")

print("")

print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
print("#"*70)
print(df.describe())
print("#"*70)

print("")
# TRAINING 
y = df.PHENO
X = df.drop(columns=['PHENO'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70:30
IDs_train = X_train.ID
IDs_test = X_test.ID
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

# Training the actual model
model = train(run_prefix, X_train, X_test, y_train, y_test, IDs_train, IDs_test)

results = model.compete() # Returns log_table 
winner = model.winner() # Returns best_algo
exported = model.export_model() # Returns algo
model.AUC()
model.export_prob_hist()
rfe = model.feature_ranking()

# TUNING
# Create a dialogue with the user 
print("Here is some basic info on the command you are about to run.")
print("CLI argument info...")
print("Working with the dataset and best model corresponding to prefix", args.prefix, "the timestamp from the merge is the prefix in most cases.")
print("Your maximum number of tuning iterations is", args.max_tune, "and if you are concerned about runtime, make this number smaller.")
print("You are running", args.n_cv, "rounds of cross-validation, and again... if you are concerned about runtime, make this number smaller.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

print("")

y_tune = df.PHENO
X_tune = df.drop(columns=['PHENO'])
IDs_tune = X_tune.ID
X_tune = X_tune.drop(columns=['ID'])

best_algo_name_in = run_prefix + '.best_algorithm.txt'
best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
best_algo = str(best_algo_df.iloc[0,0])
max_iter = args.max_tune
cv_count = args.n_cv

# Communicate to the user the best identified algorithm 
print("From previous analyses in the training phase, we've determined that the best algorithm for this application is", best_algo, " ... so let's tune it up and see what gains we can make!")

# Tuning 
model_tune = tune(run_prefix, X_tune, y_tune, IDs_tune, max_iter, cv_count)
model_tune.select_tuning_parameters(winner)
model_tune.apply_tuning_parameters()
model_tune.report_tune(rand_search.cv_results_)
model_tune.summarize_tune()
model_tune.compare_performance()
model_tune.ROC()  
model_tune.export_tuned_data()
model_tune.export_tune_hist_prob()

print()
print("Let's shut everything down, thanks for trying to tune your model with GenoML.")
print()