#! /usr/bin/env python -u
# coding=utf-8
import argparse

__author__ = 'Sayed Hadi Hashemi'
"""
Replacing trainDisc.R
Input:
prefix = args[6]                -> prune_prefix
ncores = args[7]                -> REMOVED
trainSpeed = args[8]            -> REMOVED
cvReps = args[9]                -> REMOVED
gridSearch = args[10]           -> REMOVED
imputeMissingData = args[11]    -> impute_data
NEW                             -> rank_features
output:
_bestModel.RData                -> best_model.pickle
_newModel.RData                 -> REMOVED
.nzv                            -> REMOVED
.highCor                        -> REMOVED
NEW                             -> metadata.json
_methodPerformance.tab          -> .training_withheldSamples_performanceMetrics.csv
_bestModel.algorithm            -> .best_algorithm.txt
_plotRocNoCI.png                -> .trainedModel_withheldSample_ROC.png
_plotRoc.png                    -> REMOVED
_trainingSetPredictions.tab     -> .trainedModel_withheldSample_Predictions.csv
NEW                             -> .trainedModel_trainingSample_Predictions.csv
_plotDensity.png                -> .trainedModel_withheldSample_probabilities.png
_varImp.tab                     -> .trainedModel_trainingSample_featureImportance.csv
_methodTimings.tab              -> REMOVED
"""


def compute(*, prune_prefix, impute_data, rank_features):
    pass


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments for training a discrete model')
    parser.add_argument('--prune-prefix', type=str, default='N/A', help='Prefix of GenoML run.')
    parser.add_argument('--impute-data', type=str, default='median',
                        help='Imputation: (mean, median). Governs secondary imputation and data transformation ['
                             'default: median].')
    parser.add_argument('--rank-features', type=str, default='skip',
                        help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow '
                             'with huge numbers of features [default: skip].')

    flags = parser.parse_args()
    return flags


def main():
    flags = arg_parse()
    compute(**vars(flags))


if __name__ == "__main__":
    main()
