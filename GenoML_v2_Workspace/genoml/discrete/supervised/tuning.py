# Importing the necessary packages 
import argparse
import sys
import xgboost
import sklearn
import h5py
import pandas as pd
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randfloat
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from genoml.discrete.supervised import train

class tune():
    def __init__(self, run_prefix, X_tune, y_tune, IDs_tune, max_iter, cv_count):
        self.run_prefix = run_prefix
        self.X_tune = X_tune
        self.IDs_tune = IDs_tune
        self.y_tune = y_tune
        self.max_iter = max_iter
        self.cv_count = cv_count
        self.algorithms = [
        LogisticRegression(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        SGDClassifier(loss='modified_huber'),
        SVC(probability=True),
        MLPClassifier(),
        KNeighborsClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        BaggingClassifier(),
        XGBClassifier()
        ]
        self.log_table = None
        self.best_algo = None
        self.best_algo_name_in = None
        self.best_algo_df = None
        self.hyperparameters = None
        self.scoring_metric = None
        self.cv_tuned = None 
        self.cv_baseline = None 
        self.algo = None

    def select_tuning_parameters(self, best_algo):
        best_algo = self.best_algo
                
        if best_algo == 'LogisticRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        if  best_algo == 'SGDClassifier':
            algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

        if (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
            algo = getattr(sklearn.ensemble, best_algo)()

        if best_algo == 'SVC':
            algo = getattr(sklearn.svm, best_algo)(probability=True, gamma='auto')

        if best_algo == 'ComplementNB':
            algo = getattr(sklearn.naive_bayes, best_algo)()

        if best_algo == 'MLPClassifier':
            algo = getattr(sklearn.neural_network, best_algo)()

        if best_algo == 'XGBClassifier':
            algo = getattr(xgboost, best_algo)()

        if best_algo == 'KNeighborsClassifier':
            algo = getattr(sklearn.neighbors, best_algo)()

        if (best_algo == 'LinearDiscriminantAnalysis') or (best_algo == 'QuadraticDiscriminantAnalysis'):
            algo = getattr(sklearn.discriminant_analysis, best_algo)()
        
        return algo

    def apply_tuning_parameters(self):
        # Randomized search with CV to tune
        print("Here is a summary of the top 10 iterations of the hyperparameter tuning...")
        rand_search = RandomizedSearchCV(estimator=algo, param_distributions=self.hyperparameters, scoring=self.scoring_metric, n_iter=self.max_iter, cv=self.cv_count, n_jobs=-1, random_state=153, verbose=0)
        start = time()
        rand_search.fit(self.X_tune, self.y_tune)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter iterations." % ((time() - start), self.max_iter))

    def report_tune(self, results, n_top=10):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def summarize_tune(self):
        cv_tuned = self.cv_tuned 
        cv_baseline = self.cv_baseline 

        print("Here is the cross-validation summary of your best tuned model hyperparameters...")
        cv_tuned = cross_val_score(estimator=rand_search.best_estimator_, X=self.X_tune, y=self.y_tune, scoring=self.scoring_metric, cv=self.cv_count, n_jobs=-1, verbose=0)
        print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
        print(cv_tuned)
        print("Mean cross-validation score:")
        print(cv_tuned.mean())
        print("Standard deviation of the cross-validation score:")
        print(cv_tuned.std())

        print("")

        print("Here is the cross-validation summary of your baseline/default hyperparamters for the same algorithm on the same data...")
        cv_baseline = cross_val_score(estimator=self.algo, X=self.X_tune, y=self.y_tune, scoring=self.scoring_metric, cv=self.cv_count, n_jobs=-1, verbose=0)
        print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
        print(cv_baseline)
        print("Mean cross-validation score:")
        print(cv_baseline.mean())
        print("Standard deviation of the cross-validation score:")
        print(cv_baseline.std())
        
        print()
        print("Just a note, if you have a relatively small variance among the cross-validation iterations, there is a higher chance of your model being more generalizable to similar datasets.")

    def compare_performance(self):
        print()
        if cv_baseline.mean() > cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
            print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
            print()
            print("Let's shut everything down, thanks for trying to tune your model with GenoML.")

        if cv_baseline.mean() < cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
            print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
            print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")
            
            algo_tuned = rand_search.best_estimator_
            
            ### Save it using joblib
            from joblib import dump, load
            algo_tuned_out = run_prefix + '.tunedModel.joblib'
            dump(algo_tuned, algo_tuned_out)
    
    def ROC(self):
        ### Export the ROC curve
        plot_out = self.plot_out
        fpr = self.fpr
        tpr = self.tpr 
        test_predictions = self.test_predictions 
        roc_auc = self.roc_auc 
        algo_tuned = self.algo_tuned 

        plot_out = run_prefix + '.tunedModel_allSample_ROC.png'

        test_predictions = algo_tuned.predict_proba(X_tune)
        test_predictions = test_predictions[:, 1]

        fpr, tpr, thresholds = roc_curve(y_tune, test_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % roc_auc + '\nMean cross-validation ROC curve (area = %0.2f)' % cv_tuned.mean())
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic (ROC) - ' + self.best_algo + '- tuned' )
        plt.legend(loc="lower right")
        plt.savefig(plot_out, dpi = 600)

        print()
        print("We are also exporting a ROC curve for you here", plot_out, "this is a graphical representation of AUC in all samples for the best performing algorithm.")
    
    def export_tuned_data(self):
        tune_predicteds_probs = algo_tuned.predict_proba(X_tune)
        tune_case_probs = tune_predicteds_probs[:, 1]
        tune_predicted_cases = algo_tuned.predict(X_tune)

        tune_case_probs_df = pd.DataFrame(tune_case_probs)
        tune_predicted_cases_df = pd.DataFrame(tune_predicted_cases)
        y_tune_df = pd.DataFrame(y_tune)
        IDs_tune_df = pd.DataFrame(IDs_tune)

        tune_out = pd.concat([IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_case_probs_df.reset_index(drop=True), tune_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        tune_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        tune_out = tune_out.drop(columns=['INDEX'])

        tune_outfile = run_prefix + '.tunedModel_allSample_Predictions.csv'
        tune_out.to_csv(tune_outfile, index=False)

        print("")
        print("Preview of the exported predictions for the tuning samples which is naturally overfit and exported as", tune_outfile, "in the similar format as in the initial training phase of GenoML.")
        print("#"*70)
        print(tune_out.head())
        print("#"*70)
    
    def export_tune_hist_prob(self):
        genoML_colors = ["cyan","purple"]

        g = sns.FacetGrid(tune_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
        g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=False, rug=True))
        g.add_legend()

        plot_out = run_prefix + '.tunedModel_allSample_probabilities.png'
        g.savefig(plot_out, dpi=600)

        print("")
        print("We are also exporting probability density plots to the file", plot_out, "this is a plot of the probability distributions of being a case, stratified by case and control status for all samples.")

