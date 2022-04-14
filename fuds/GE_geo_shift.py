# The evaluation of the performance of the Gerry Fair classifier for different states:


# The packages to download:

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

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

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


# We perform the evaluation for each state:

state = sys.argv[1]

feat = ['COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P']

    
FPR_GE = np.array([])
TPR_GE = np.array([])
PPV_GE = np.array([])
NPV_GE = np.array([])
ACC_GE = np.array([])
SP_GE = np.array([])

max_iterations = 500
C = 100
print_flag = False
gamma = .01

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
dataset_orig_train = data_all
    
fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FN',
                                 max_iters=max_iterations, heatmapflag=False)
    
fair_model.fit(dataset_orig_train, early_termination=True)
    
for state2 in state_list:
    if (state2 != state):
            
        acs_data_test = data_source.get_data(states=[str(state2)], download=True)
        data_test = acs_data_test[feat]
        features, label, group = ACSEmployment.df_to_numpy(acs_data_test)
        # stick to instances with no NAN values
        data_test = data_test.dropna()
        index = data_test.index
        a_list = list(index)
        new_label = label[a_list]
        data_test['label'] = new_label
        favorable_classes = [True]
        protected_attribute_names = ['SEX']
        privileged_classes = np.array([[1]])
        data_all_test = StandardDataset(data_test, 'label', favorable_classes = favorable_classes,
                                        protected_attribute_names = protected_attribute_names,
                                        privileged_classes = privileged_classes)
        privileged_groups = [{'SEX': 1}]
        unprivileged_groups = [{'SEX': 2}]
        dataset_orig_test = data_all_test
    
            
        dataset_debiasing_test = fair_model.predict(dataset_orig_test, threshold=0.5)

    
        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_debiasing_test,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
        fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
        tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
        ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
        npv = cm_transf_test.difference(cm_transf_test.negative_predictive_value)
        acc = cm_transf_test.accuracy()
        sp = cm_transf_test.statistical_parity_difference()

        FPR_GE = np.append(FPR_GE, fpr)
        TPR_GE = np.append(TPR_GE, tpr)
        PPV_GE = np.append(PPV_GE, ppv)
        NPV_GE = np.append(NPV_GE, npv)
        ACC_GE = np.append(ACC_GE, acc)
        SP_GE = np.append(SP_GE, sp)
                      
        

filename = "Adult_geo_gender_GE_geo_shift_"+ state + ".txt"

a_file = open(filename, "w")

res = [FPR_GE, TPR_GE, PPV_GE, NPV_GE, ACC_GE, SP_GE]

for metric in res:
    np.savetxt(a_file, metric)

a_file.close()
            
            
