# Evaluation of the performance of the Kamiran algorithm (ethicML)

# TIME SHIFT

# import the required packages
import nest_asyncio

from fuds.utilties import model_seed

nest_asyncio.apply()

from pathlib import Path

import ethicml as em
import numpy as np


def main():
    root_dir = Path("..")

    # States list can from
    state_list_short = ["CA", "AK", "HI", "KS", "NE", "ND", "NY", "OR", "PR", "TX", "VT", "WY"]

    for state in state_list_short:

        FPR = np.array([])
        FNR = np.array([])
        TPR = np.array([])
        PPV = np.array([])
        ACC = np.array([])
        GACC = np.array([])
        DI = np.array([])
        NPV = np.array([])

        data: em.DataTuple = em.acs_income(
            root=root_dir, year=2014, states=[state], horizon=1
        ).load()

        for data_seed in range(10):  # 10-fold cross validation, save values for each fold.
            train, test = em.train_test_split(data, train_percentage=0.7, random_seed=data_seed)

            clf = em.Kamiran(seed=model_seed)
            pred = clf.run(train, test)

            metrics = [em.Accuracy()]
            per_sens = [
                em.Accuracy(),
                em.ProbPos(),
                em.TPR(),
                em.FNR(),
                em.FPR(),
                em.PPV(),
                em.NPV(),
            ]
            evaluation = em.run_metrics(pred, test, metrics, per_sens)
            FPR = np.append(FPR, evaluation["FPR_SEX_1_0-SEX_1_1"])
            FNR = np.append(FNR, evaluation["FNR_SEX_1_0-SEX_1_1"])
            TPR = np.append(TPR, evaluation["TPR_SEX_1_0-SEX_1_1"])
            PPV = np.append(PPV, evaluation["PPV_SEX_1_0-SEX_1_1"])
            ACC = np.append(ACC, evaluation["Accuracy"])
            GACC = np.append(GACC, evaluation["Accuracy_SEX_1_0-SEX_1_1"])
            DI = np.append(DI, evaluation["prob_pos_SEX_1_0/SEX_1_1"])
            NPV = np.append(NPV, evaluation["NPV_SEX_1_0-SEX_1_1"])

        filename = f"Adult_time_gender_Kamiran_eval_{state}.txt"

        with open(filename, "w") as a_file:
            res = [FPR, FNR, TPR, PPV, NPV, ACC, GACC, DI]

            for metric in res:
                np.savetxt(a_file, metric)


if __name__ == "__main__":
    main()
