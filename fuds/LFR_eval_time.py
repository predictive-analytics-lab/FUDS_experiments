# The evaluation of the performance of the LFR algorithm for different states:

# TIME SHIFT


# The packages to download:


from aif360.algorithms.preprocessing import LFR
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource
import numpy as np
from sklearn.preprocessing import StandardScaler

from fuds.utilties import load_acs_aif, model_seed, state_list_short


def main():
    # Load the data:
    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
    # We perform the evaluation for each state:
    for state in state_list_short:

        FPR_LFR = np.array([])
        FNR_LFR = np.array([])
        TPR_LFR = np.array([])
        PPV_LFR = np.array([])
        FOR_LFR = np.array([])
        ACC_LFR = np.array([])

        dataset_orig, privileged_groups, unprivileged_groups = load_acs_aif(data_source, state)

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split(
                [0.7], shuffle=True, seed=data_seed
            )

            scale_orig = StandardScaler()
            dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)

            TR = LFR(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                k=10,
                Ax=0.1,
                Ay=1.0,
                Az=2.0,
                verbose=1,
                seed=model_seed,
            )
            TR = TR.fit(dataset_orig_train, maxiter=5000, maxfun=5000)

            TR.transform(dataset_orig_train)

            # test the classifier:

            dataset_orig_test.features = scale_orig.transform(dataset_orig_test.features)
            dataset_transf_test = TR.transform(dataset_orig_test)

            dataset_transf_test_new = dataset_orig_test.copy(deepcopy=True)
            dataset_transf_test_new.scores = dataset_transf_test.scores

            fav_inds = dataset_transf_test_new.scores > 0.5
            dataset_transf_test_new.labels[fav_inds] = 1.0
            dataset_transf_test_new.labels[~fav_inds] = 0.0

            cm_transf_test = ClassificationMetric(
                dataset_orig_test,
                dataset_transf_test_new,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
            fnr = cm_transf_test.difference(cm_transf_test.false_negative_rate)
            tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
            ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
            fom = cm_transf_test.difference(cm_transf_test.false_omission_rate)
            acc = cm_transf_test.accuracy()

            FPR_LFR = np.append(FPR_LFR, fpr)
            FNR_LFR = np.append(FNR_LFR, fnr)
            TPR_LFR = np.append(TPR_LFR, tpr)
            PPV_LFR = np.append(PPV_LFR, ppv)
            FOR_LFR = np.append(FOR_LFR, fom)
            ACC_LFR = np.append(ACC_LFR, acc)

        filename = f"Adult_time_gender_LFR_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_LFR, FNR_LFR, TPR_LFR, PPV_LFR, FOR_LFR, ACC_LFR]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
