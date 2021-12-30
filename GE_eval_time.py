# The evaluation of the performance of the Gerry Fair classifier for different states:

# TIME SHIFT

state_list_short = ['CA', 'AK', 'HI', 'KS', 'NE', 'ND', 'NY', 'OR',
                    'PR', 'TX', 'VT', 'WY']


# The packages to download:

import sys
import numpy as np
import pandas as pd

from aif360.algorithms.inprocessing import GerryFairClassifier

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.datasets import StandardDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow.compat.v1 as tf


from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

from folktables import ACSDataSource, ACSEmployment

# Load the data:

data_source = ACSDataSource(survey_year='2014', horizon='1-Year', survey='person')


# We perform the evaluation for each state:



for state in state_list_short:
    
    FPR_GE = np.array([])
    FNR_GE = np.array([])
    TPR_GE = np.array([])
    PPV_GE = np.array([])
    FOR_GE = np.array([])
    ACC_GE = np.array([])

    max_iterations = 500
    C = 100
    print_flag = False
    gamma = .005

    acs_data = data_source.get_data(states=[state], download=True)
    data = acs_data[feat]
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    # stick to instances with no NAN values
    data = data.dropna()
    index = data.index
    a_list = list(index)
    new_label = label[a_list]
    data['label'] = new_label
    favorable_classes = [True]
    protected_attribute_names = ['SEX']
    privileged_classes = np.array([[1]])
    data_all = StandardDataset(data, 'label', favorable_classes = favorable_classes, 
                         protected_attribute_names = protected_attribute_names,
                        privileged_classes = privileged_classes)
    privileged_groups = [{'SEX': 1}]
    unprivileged_groups = [{'SEX': 2}]
    dataset_orig = data_all
    
    for i in range(10): #10-fold cross validation, save values for each fold.
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

        fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                                         max_iters=max_iterations, heatmapflag=False)
        fair_model.fit(dataset_orig_train, early_termination=True)

        # test the classifier:

        dataset_debiasing_test = fair_model.predict(dataset_orig_test, threshold=0.5)

    
        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_debiasing_test,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
        fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
        fnr = cm_transf_test.difference(cm_transf_test.false_negative_rate)
        tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
        ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
        fom = cm_transf_test.difference(cm_transf_test.false_omission_rate)
        acc = cm_transf_test.accuracy()
            
        FPR_GE = np.append(FPR_GE, fpr)
        FNR_GE = np.append(FNR_GE, fnr)
        TPR_GE = np.append(TPR_GE, tpr)
        PPV_GE = np.append(PPV_GE, ppv)
        FOR_GE = np.append(FOR_GE, fom)
        ACC_GE = np.append(ACC_GE, acc)
            
        

    filename = "Adult_time_gender_GE_eval_"+ state + ".txt"

    a_file = open(filename, "w")

    res = [FPR_GE, FNR_GE, TPR_GE, PPV_GE, FOR_GE, ACC_GE]

    for metric in res:
        np.savetxt(a_file, metric)

    a_file.close()
            
            
