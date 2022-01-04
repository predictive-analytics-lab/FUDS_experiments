# The evaluation of the performance of the CEO post-processing classifier for different states:

# TIME SHIFT


# The packages to download:


from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from folktables import ACSDataSource
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utilties import load_acs_aif, model_seed, state_list_short


def main():
    # Load the data
    data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")

    class_thresh = 0.5
    # cost constraint of fnr will optimize generalized false negative rates, that of
    # fpr will optimize generalized false positive rates, and weighted will optimize
    # a weighted combination of both
    cost_constraint = "weighted"  # "fnr", "fpr", "weighted"
    # random seed for calibrated equal odds prediction

    # We perform the evaluation for each state:

    for state in state_list_short:

        FPR_CEO = np.array([])
        FNR_CEO = np.array([])
        TPR_CEO = np.array([])
        PPV_CEO = np.array([])
        FOR_CEO = np.array([])
        ACC_CEO = np.array([])

        dataset_orig, privileged_groups, unprivileged_groups = load_acs_aif(data_source, state)

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

            dataset_orig_train, dataset_orig_vt = dataset_orig.split(
                [0.6], shuffle=True, seed=data_seed
            )
            dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split(
                [0.5], shuffle=True, seed=data_seed
            )

            dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
            dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
            dataset_new_valid_pred = dataset_orig_valid.copy(
                deepcopy=True
            )  # This variable is unused

            # train the Logistic Regression Model

            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            lmod = LogisticRegression()
            lmod.fit(X_train, y_train)

            fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
            y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

            # Prediction probs for validation and testing data
            X_valid = scale_orig.transform(dataset_orig_valid.features)
            y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

            dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
            dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)

            y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
            y_train_pred[
                y_train_pred_prob >= class_thresh
            ] = dataset_orig_train_pred.favorable_label
            y_train_pred[
                ~(y_train_pred_prob >= class_thresh)
            ] = dataset_orig_train_pred.unfavorable_label
            dataset_orig_train_pred.labels = y_train_pred

            y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
            y_valid_pred[
                y_valid_pred_prob >= class_thresh
            ] = dataset_orig_valid_pred.favorable_label
            y_valid_pred[
                ~(y_valid_pred_prob >= class_thresh)
            ] = dataset_orig_valid_pred.unfavorable_label
            dataset_orig_valid_pred.labels = y_valid_pred

            # Learn parameters to equalize odds and apply to create a new dataset
            cpp = CalibratedEqOddsPostprocessing(
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups,
                cost_constraint=cost_constraint,
                seed=model_seed,
            )

            cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

            # test the classifier:

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)  # This variable is unused

            X_test = scale_orig.transform(dataset_orig_test.features)
            y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

            dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

            y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
            y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
            y_test_pred[
                ~(y_test_pred_prob >= class_thresh)
            ] = dataset_orig_test_pred.unfavorable_label
            dataset_orig_test_pred.labels = y_test_pred

            dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

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

            FPR_CEO = np.append(FPR_CEO, fpr)
            FNR_CEO = np.append(FNR_CEO, fnr)
            TPR_CEO = np.append(TPR_CEO, tpr)
            PPV_CEO = np.append(PPV_CEO, ppv)
            FOR_CEO = np.append(FOR_CEO, fom)
            ACC_CEO = np.append(ACC_CEO, acc)

        filename = f"Adult_time_gender_CEO_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR_CEO, FNR_CEO, TPR_CEO, PPV_CEO, FOR_CEO, ACC_CEO]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
