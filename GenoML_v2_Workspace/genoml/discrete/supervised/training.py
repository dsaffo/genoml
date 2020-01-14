# Importing the necessary packages 
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import sklearn
import time
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE

# Initialize training class
class train:
    def __init__(self, run_prefix, X_train, X_test, y_train, y_test, IDs_train, IDs_test):
        self.run_prefix = run_prefix
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.IDs_train = IDs_train
        self.IDs_test = IDs_test
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
        self.algo = None


    def compete(self):
        print("")
        print("Now let's compete these algorithms!")
        print("We'll update you as each algorithm runs, then summarize at the end.")
        print("Here we test each algorithm under default settings using the same training and test datasets derived from a 70% training and 30% testing split of your data.")
        print("For each algorithm, we will output the following metrics...")
        print("Algorithm name, hoping that's pretty self-explanatory. Plenty of resources on these common ML algorithms at https://scikit-learn.org and https://xgboost.readthedocs.io/.")
        print("AUC_percent, this is the area under the curve from receiver operating characteristic analyses. This is the most common metric of classifier performance in biomedical literature, we express this as a percent. We calculate AUC based on the predicted probability of being a case.")
        print("Accuracy_percent, this is the simple accuracy of the classifier, how many predictions were correct from best classification cutoff (python default).")
        print("Balanced_Accuracy_Percent, consider this as the accuracy resampled to a 1:1 mix of cases and controls. Imbalanced datasets can give funny results for simple accuracy.")
        print("Log_Loss, this is essentially the inverse of the likelihood function for a correct prediction, you want to minimize this.")
        print("Sensitivity, proportion of cases correctly identified.")
        print("Specificity, proportion of controls correctly identified.")
        print("PPV, this is the positive predictive value, the probability that subjects with a positive result actually have the disease.")
        print("NPV, this is the negative predictive value, the probability that subjects with a negative result don't have the disease.")
        print("We also log the runtimes per algorithm.")

        print("")

        print("Algorithm summaries incoming...")

        print("")

        log_cols=["Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
        log_table = pd.DataFrame(columns=log_cols)

        for algo in self.algorithms:
            
            start_time = time.time()
            
            algo.fit(self.X_train, self.y_train)
            name = algo.__class__.__name__

            print("")
            print("#"*70)
            print("")
            print(name)

            test_predictions = algo.predict_proba(self.X_test)
            test_predictions = test_predictions[:, 1]
            rocauc = roc_auc_score(self.y_test, test_predictions)
            print("AUC: {:.4%}".format(rocauc))

            test_predictions = algo.predict(self.X_test)
            acc = accuracy_score(self.y_test, test_predictions)
            print("Accuracy: {:.4%}".format(acc))

            test_predictions = algo.predict(self.X_test)
            balacc = balanced_accuracy_score(self.y_test, test_predictions)
            print("Balanced Accuracy: {:.4%}".format(balacc))
            
            CM = confusion_matrix(self.y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP)
            PPV = TP/(TP+FP)
            NPV = TN/(TN+FN)
            

            test_predictions = algo.predict_proba(self.X_test)
            ll = log_loss(self.y_test, test_predictions)
            print("Log Loss: {:.4}".format(ll))
            
            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("Runtime in seconds: {:.4}".format(elapsed_time))

            log_entry = pd.DataFrame([[name, rocauc*100, acc*100, balacc*100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=log_cols)
            log_table = log_table.append(log_entry)

        print("#"*70)

        print("")

        log_outfile = self.run_prefix + '.training_withheldSamples_performanceMetrics.csv'

        print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
        print("#"*70)
        print(log_table)
        print("#"*70)

        log_table.to_csv(log_outfile, index=False)

        self.log_table = log_table

        return log_table

    def winner(self):
        best_performing_summary = self.log_table[self.log_table.AUC_Percent == self.log_table.AUC_Percent.max()]
        best_algo = best_performing_summary.at[0,'Algorithm']

        print("")

        print(f"Based on your withheld samples, the algorithm with the best AUC is the {best_algo}... let's save that model for you.")

        best_algo_name_out = self.run_prefix + ".best_algorithm.txt"
        file = open(best_algo_name_out,'w')
        file.write(best_algo)
        file.close() 

        self.best_algo = best_algo

        return best_algo

    def export_model(self):
        best_algo = self.best_algo

        if best_algo == 'LogisticRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        if  best_algo == 'SGDClassifier':
            algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

        if (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
            algo = getattr(sklearn.ensemble, best_algo)()

        if best_algo == 'SVC':
            algo = getattr(sklearn.svm, best_algo)(probability=True)

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

        algo.fit(self.X_train, self.y_train)
        name = algo.__class__.__name__

        print("...remember, there are occasionally slight fluctuations in model performance on the same withheld samples...")

        print("#"*70)

        print(name)

        test_predictions = algo.predict_proba(self.X_test)
        test_predictions = test_predictions[:, 1]
        rocauc = roc_auc_score(self.y_test, test_predictions)
        print("AUC: {:.4%}".format(rocauc))

        test_predictions = algo.predict(self.X_test)
        acc = accuracy_score(self.y_test, test_predictions)
        print("Accuracy: {:.4%}".format(acc))

        test_predictions = algo.predict(self.X_test)
        balacc = balanced_accuracy_score(self.y_test, test_predictions)
        print("Balanced Accuracy: {:.4%}".format(balacc))

        test_predictions = algo.predict_proba(self.X_test)
        ll = log_loss(self.y_test, test_predictions)
        print("Log Loss: {:.4}".format(ll))

        ### Save it using joblib
       
        algo_out = self.run_prefix + '.trainedModel.joblib'
        dump(algo, algo_out)

        print("#"*70)

        print(f"... this model has been saved as {algo_out} for later use and can be found in your working directory.")

        self.algo = algo

        return algo

    def AUC(self):
        plot_out = self.run_prefix + '.trainedModel_withheldSample_ROC.png'

        test_predictions = self.algo.predict_proba(self.X_test)
        test_predictions = test_predictions[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y_test, test_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic (ROC) - ' + self.best_algo)
        plt.legend(loc="lower right")
        plt.savefig(plot_out, dpi = 600)

        print()
        print(f"We are also exporting a ROC curve for you here {plot_out} this is a graphical representation of AUC in the withheld test data for the best performing algorithm.")

    def export_prob_hist(self):
        # Exporting withheld test data
        test_predicteds_probs = self.algo.predict_proba(self.X_test)
        test_case_probs = test_predicteds_probs[:, 1]
        test_predicted_cases = self.algo.predict(self.X_test)

        test_case_probs_df = pd.DataFrame(test_case_probs)
        test_predicted_cases_df = pd.DataFrame(test_predicted_cases)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_case_probs_df.reset_index(drop=True), test_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        test_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        test_out = test_out.drop(columns=['INDEX'])

        test_outfile = self.run_prefix + '.trainedModel_withheldSample_Predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print("Preview of the exported predictions for the withheld test data that has been exported as", test_outfile, "these are pretty straight forward.")
        print("They generally include the sample ID, the previously reported case status (1 = case), the case probability from the best performing algorithm and the predicted label from that algorithm")
        print("")
        print("#"*70)
        print(test_out.head())
        print("#"*70)


        # Exporting training data, which is by nature overfit.
        train_predicteds_probs = self.algo.predict_proba(self.X_train)
        train_case_probs = train_predicteds_probs[:, 1]
        train_predicted_cases = self.algo.predict(self.X_train)

        train_case_probs_df = pd.DataFrame(train_case_probs)
        train_predicted_cases_df = pd.DataFrame(train_predicted_cases)
        y_train_df = pd.DataFrame(self.y_train)
        IDs_train_df = pd.DataFrame(self.IDs_train)

        train_out = pd.concat([IDs_train_df.reset_index(), y_train_df.reset_index(drop=True), train_case_probs_df.reset_index(drop=True), train_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        train_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        train_out = train_out.drop(columns=['INDEX'])

        train_outfile = self.run_prefix + '.trainedModel_trainingSample_Predictions.csv'
        train_out.to_csv(train_outfile, index=False)

        print("")
        print("Preview of the exported predictions for the training samples which is naturally overfit and exported as", train_outfile, "in the similar format as in the withheld test dataset that was just exported.")
        print("#"*70)
        print(train_out.head())
        print("#"*70)

        # Export historgrams of probabilities
        genoML_colors = ["cyan","purple"]

        g = sns.FacetGrid(train_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
        g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=False, rug=True))
        g.add_legend()

        plot_out = self.run_prefix + '.trainedModel_withheldSample_probabilities.png'
        g.savefig(plot_out, dpi=600)

        print("")
        print("We are also exporting probability density plots to the file", plot_out, "this is a plot of the probability distributions of being a case, stratified by case and control status in the withheld test samples.")

    def feature_ranking(self):
        best_algo = self.best_algo
        X_train = self.X_train
        y_train = self.y_train
        if (best_algo == 'SVC') or (best_algo == 'ComplementNB') or (best_algo == 'KNeighborsClassifier') or (best_algo == 'QuadraticDiscriminantAnalysis') or (best_algo == 'BaggingClassifier'):
        
            print("Even if you selected to run feature ranking, you can't generate feature ranks using SVC, ComplementNB, KNeighborsClassifier, QuadraticDiscriminantAnalysis, or BaggingClassifier... it just isn't possible.")
        
        else:
            print("Processing feature ranks, this can take a while. But you will get a relative rank for every feature in the model.")

            top_ten_percent = (len(X_train)//10)
            # core_count = args.n_cores
            names = list(X_train.columns)
            rfe = RFE(estimator=self.algo)
            rfe.fit(X_train, y_train)
            rfe_out = zip(rfe.ranking_, names)
            rfe_df = pd.DataFrame(rfe_out, columns = ["RANK","FEATURE"])
            table_outfile = self.run_prefix + '.trainedModel_trainingSample_featureImportance.csv'
            rfe_df.to_csv(table_outfile, index=False)

            return rfe_df
        
            print(f"Feature ranks exported as {table_outfile} if you want to be very picky and make a more parsimonious model with a minimal feature set, extract all features ranked 1 and rebuild your dataset. This analysis also gives you a concept of the relative importance of your features in the model.")
