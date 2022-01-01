# The evaluation of the performance of the Logistic Regression classifier for different states:

# TIME SHIFT


# The packages to download:


from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource, ACSEmployment
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    # Load the data:
    state_list_short = [
        "CA",
        "AK",
        "HI",
        "KS",
        "NE",
        "ND",
        "NY",
        "OR",
        "PR",
        "TX",
        "VT",
        "WY",
    ]
    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
    
    feat = ['COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P']

    # We perform the evaluation for each state:

    for state in state_list_short:

        FPR_LR = np.array([])
        FNR_LR = np.array([])
        TPR_LR = np.array([])
        PPV_LR = np.array([])
        FOR_LR = np.array([])
        ACC_LR = np.array([])

        acs_data = data_source.get_data(states=[state], download=True)
        data = acs_data[feat]
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
        # stick to instances with no NAN values
        data = data.dropna()
        index = data.index
        a_list = list(index)
        new_label = label[a_list]
        data["label"] = new_label
        favorable_classes = [True]
        protected_attribute_names = ["SEX"]
        privileged_classes = np.array([[1]])
        data_all = StandardDataset(
            data,
            "label",
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
        )
        privileged_groups = [{"SEX": 1}]
        unprivileged_groups = [{"SEX": 2}]
        dataset_orig = data_all

        for _ in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

            dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

            # train the Logistic Regression Model

            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            lmod = LogisticRegression()
            lmod.fit(X_train, y_train)

            # test the classifier:

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

            X_test = scale_orig.transform(dataset_orig_test.features)
            y_test_pred_prob = lmod.predict_proba(X_test)[:, 1]

            dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

            y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
            y_test_pred[y_test_pred_prob >= 0.5] = 1.0
            y_test_pred[~(y_test_pred_prob >= 0.5)] = 0.0
            dataset_orig_test_pred.labels = y_test_pred

            cm_transf_test = ClassificationMetric(
                dataset_orig_test,
                dataset_orig_test_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
            fnr = cm_transf_test.difference(cm_transf_test.false_negative_rate)
            tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
            ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
            fom = cm_transf_test.difference(cm_transf_test.false_omission_rate)
            acc = cm_transf_test.accuracy()

            FPR_LR = np.append(FPR_LR, fpr)
            FNR_LR = np.append(FNR_LR, fnr)
            TPR_LR = np.append(TPR_LR, tpr)
            PPV_LR = np.append(PPV_LR, ppv)
            FOR_LR = np.append(FOR_LR, fom)
            ACC_LR = np.append(ACC_LR, acc)

        filename = "Adult_time_gender_LR_eval_" + state + ".txt"

        with open(filename, "w") as a_file:
            res = [FPR_LR, FNR_LR, TPR_LR, PPV_LR, FOR_LR, ACC_LR]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
