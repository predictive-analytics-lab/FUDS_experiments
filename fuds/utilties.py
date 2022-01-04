from typing import Dict, List, Tuple

from aif360.datasets import StandardDataset
from folktables import ACSDataSource, ACSEmployment
import numpy as np

model_seed = 12345679
state_list_short = ["CA", "AK", "HI", "KS", "NE", "ND", "NY", "OR", "PR", "TX", "VT", "WY"]
feat = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']


def load_acs_aif(
    data_source: ACSDataSource, state: str
) -> Tuple[StandardDataset, List[Dict[str, int]], List[Dict[str, int]]]:
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
    return dataset_orig, privileged_groups, unprivileged_groups
