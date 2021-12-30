# The evaluation of the performance of the EO post-processing classifier for different states:

state_list_short = ['CA', 'AK', 'HI', 'KS', 'NE', 'ND', 'NY', 'OR',
                    'PR', 'TX', 'VT', 'WY']


# The packages to download:

import sys
import numpy as np
import pandas as pd

from aif360.algorithms.postprocessing import EqOddsPostprocessing

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

# Load the data

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')


# We perform the evaluation for each state:



for state in state_list_short:
    
    FPR_EO = np.array([])
    FNR_EO = np.array([])
    TPR_EO = np.array([])
    PPV_EO = np.array([])
    FOR_EO = np.array([])
    ACC_EO = np.array([])

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
        #dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    
        # train the Logistic Regression Model
    
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
    
        fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        y_train_pred_prob = lmod.predict_proba(X_train)[:,fav_idx]

        # Prediction probs for validation and testing data
        X_valid = scale_orig.transform(dataset_orig_valid.features)
        y_valid_pred_prob = lmod.predict_proba(X_valid)[:,fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
    

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred
    
     

        # Learn parameters to equalize odds and apply to create a new dataset
        cpp = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                             unprivileged_groups = unprivileged_groups)
    
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

        # test the classifier:

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)
    
        X_test = scale_orig.transform(dataset_orig_test.features)
        y_test_pred_prob = lmod.predict_proba(X_test)[:,fav_idx]
    
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)
    
        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred
    
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
    
        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
        fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
        fnr = cm_transf_test.difference(cm_transf_test.false_negative_rate)
        tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
        ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
        fom = cm_transf_test.difference(cm_transf_test.false_omission_rate)
        acc = cm_transf_test.accuracy()
            
        FPR_EO = np.append(FPR_EO, fpr)
        FNR_EO = np.append(FNR_EO, fnr)
        TPR_EO = np.append(TPR_EO, tpr)
        PPV_EO = np.append(PPV_EO, ppv)
        FOR_EO = np.append(FOR_EO, fom)
        ACC_EO = np.append(ACC_EO, acc)
            
        

    filename = "Adult_geo_gender_EO_eval_"+ state + ".txt"

    a_file = open(filename, "w")

    res = [FPR_EO, FNR_EO, TPR_EO, PPV_EO, FOR_EO, ACC_EO]

    for metric in res:
        np.savetxt(a_file, metric)

    a_file.close()
            
            
