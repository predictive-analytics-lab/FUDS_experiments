# The evaluation of the performance of the RW algorithm for different states:

# TIME SHIFT


# The packages to download:


from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource, ACSEmployment
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fuds.utilties import model_seed, state_list_short, feat


def main():
    # Load the data

    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")

    # We perform the evaluation for each state:

    for state in state_list_short:

        FPR_RW = np.array([])
        FNR_RW = np.array([])
        TPR_RW = np.array([])
        PPV_RW = np.array([])
        FOR_RW = np.array([])
        ACC_RW = np.array([])

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

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            dataset_orig_train, dataset_orig_test = dataset_orig.split(
                [0.7], shuffle=True, seed=data_seed
            )

            RW = Reweighing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            RW.fit(dataset_orig_train)
            dataset_transf_train = RW.transform(dataset_orig_train)

            scale_transf = StandardScaler()
            X_train = scale_transf.fit_transform(dataset_transf_train.features)
            y_train = dataset_transf_train.labels.ravel()

            rand_state = np.random.RandomState(model_seed)
            lmod = LogisticRegression(random_state=rand_state)
            lmod.fit(X_train, y_train, sample_weight=dataset_transf_train.instance_weights)
            lmod.predict(X_train)

            # test the classifier:

            pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

            dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
            X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
            dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

            fav_inds = dataset_transf_test_pred.scores > 0.5
            dataset_transf_test_pred.labels[fav_inds] = 1.0
            dataset_transf_test_pred.labels[~fav_inds] = 0.0

            cm_transf_test = ClassificationMetric(
                dataset_orig_test,
                dataset_transf_test_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            fpr = cm_transf_test.difference(cm_transf_test.false_positive_rate)
            fnr = cm_transf_test.difference(cm_transf_test.false_negative_rate)
            tpr = cm_transf_test.difference(cm_transf_test.true_positive_rate)
            ppv = cm_transf_test.difference(cm_transf_test.positive_predictive_value)
            fom = cm_transf_test.difference(cm_transf_test.false_omission_rate)
            acc = cm_transf_test.accuracy()

            FPR_RW = np.append(FPR_RW, fpr)
            FNR_RW = np.append(FNR_RW, fnr)
            TPR_RW = np.append(TPR_RW, tpr)
            PPV_RW = np.append(PPV_RW, ppv)
            FOR_RW = np.append(FOR_RW, fom)
            ACC_RW = np.append(ACC_RW, acc)

        filename = f"Adult_time_gender_RW_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_RW, FNR_RW, TPR_RW, PPV_RW, FOR_RW, ACC_RW]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
