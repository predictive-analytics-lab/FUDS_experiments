# The evaluation of the performance of the Gerry Fair classifier for different states:


# The packages to download:


from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource, ACSEmployment
import numpy as np


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
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    
    feat = ['COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P']

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

        filename = "Adult_geo_gender_GE_eval_" + state + ".txt"

        with open(filename, "w") as a_file:
            res = [FPR_GE, FNR_GE, TPR_GE, PPV_GE, FOR_GE, ACC_GE]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
