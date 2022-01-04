# The evaluation of the performance of the Meta classifier for different states:


# The packages to download:


from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from utilties import load_acs_aif, model_seed, state_list_short


def main():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    # We perform the evaluation for each state:

    for state in state_list_short:

        FPR_META = np.array([])
        FNR_META = np.array([])
        TPR_META = np.array([])
        PPV_META = np.array([])
        FOR_META = np.array([])
        ACC_META = np.array([])

        dataset_orig, privileged_groups, unprivileged_groups = load_acs_aif(data_source, state)

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split(
                [0.7], shuffle=True, seed=data_seed
            )

            min_max_scaler = MaxAbsScaler()
            dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
            debiased_model = MetaFairClassifier(
                tau=0.7, sensitive_attr="SEX", type="fdr", seed=model_seed
            ).fit(dataset_orig_train)

            # test the classifier:

            dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
            dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

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

            FPR_META = np.append(FPR_META, fpr)
            FNR_META = np.append(FNR_META, fnr)
            TPR_META = np.append(TPR_META, tpr)
            PPV_META = np.append(PPV_META, ppv)
            FOR_META = np.append(FOR_META, fom)
            ACC_META = np.append(ACC_META, acc)

        filename = f"Adult_geo_gender_META_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_META, FNR_META, TPR_META, PPV_META, FOR_META, ACC_META]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
