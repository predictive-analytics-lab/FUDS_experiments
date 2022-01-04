# The evaluation of the performance of the Logistic Regression classifier for different states:


# The packages to download:


from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utilties import load_acs_aif, model_seed, state_list_short


def main():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    # We perform the evaluation for each state:
    for state in state_list_short:
        FPR_LR = np.array([])
        FNR_LR = np.array([])
        TPR_LR = np.array([])
        PPV_LR = np.array([])
        FOR_LR = np.array([])
        ACC_LR = np.array([])

        dataset_orig, privileged_groups, unprivileged_groups = load_acs_aif(data_source, state)

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split(
                [0.7], shuffle=True, seed=data_seed
            )

            dataset_orig_train_pred = dataset_orig_train.copy(
                deepcopy=True
            )  # This variable is unused

            # train the Logistic Regression Model

            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            rand_state = np.random.RandomState(model_seed)
            lmod = LogisticRegression(random_state=rand_state)
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

        filename = f"Adult_geo_gender_LR_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_LR, FNR_LR, TPR_LR, PPV_LR, FOR_LR, ACC_LR]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
