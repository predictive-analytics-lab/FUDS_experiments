# The evaluation of the performance of the Gerry Fair classifier for different states:


# The packages to download:


from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource
import numpy as np

from utilties import load_acs_aif, state_list_short


def main():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    max_iterations = 500
    C = 100
    print_flag = False
    gamma = 0.005

    # We perform the evaluation for each state:

    for state in state_list_short:

        FPR_GE = np.array([])
        FNR_GE = np.array([])
        TPR_GE = np.array([])
        PPV_GE = np.array([])
        FOR_GE = np.array([])
        ACC_GE = np.array([])

        dataset_orig, privileged_groups, unprivileged_groups = load_acs_aif(data_source, state)

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split(
                [0.7], shuffle=True, seed=data_seed
            )

            fair_model = GerryFairClassifier(
                C=C,
                printflag=print_flag,
                gamma=gamma,
                fairness_def="FP",
                max_iters=max_iterations,
                heatmapflag=False,
            )
            fair_model.fit(dataset_orig_train, early_termination=True)

            # test the classifier:

            dataset_debiasing_test = fair_model.predict(dataset_orig_test, threshold=0.5)

            cm_transf_test = ClassificationMetric(
                dataset_orig_test,
                dataset_debiasing_test,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
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

        filename = f"Adult_geo_gender_GE_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_GE, FNR_GE, TPR_GE, PPV_GE, FOR_GE, ACC_GE]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
