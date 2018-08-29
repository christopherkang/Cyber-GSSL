import numpy as np
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))


def write_label(filename):
    samp_csv = pd.read_csv(("../%s" % (filename))).shape[1]
    print(samp_csv)
    col_head = ["CVE %d" % x for x in range(1, samp_csv)]
    col_head.insert(0, "CWE")

    csv_to_be_read = pd.read_csv(
        ("../%s" % (filename)),
        header=None, names=col_head, index_col=0)

    dup_checker = []

    for row in csv_to_be_read.index.values:
        for counter in range(1, 34):
            val_to_store = csv_to_be_read.loc[(row), "CVE %d" % counter]
            if type(val_to_store) == float:
                break
            dup_checker.append([val_to_store, row])
    # new_hierarchy = pd.MultiIndex.from_arrays(dup_checker)
    dup_checker = np.asarray(dup_checker)

    real_df = pd.DataFrame(
        dup_checker[:, 0], index=np.asarray(dup_checker[:, 1], dtype=np.int),
        columns=["CVE"])

    real_df_CVE = pd.DataFrame(
        np.asarray(dup_checker[:, 1], dtype=np.int),
        index=dup_checker[:, 0], columns=["CWE"])

    real_df.to_pickle("../data/%s_CWE" % ("_".join(filename.split(".")[:-1])))
    real_df_CVE.to_pickle("../data/%s_CVE" % ("_".join(filename.split(".")[:-1])))

write_label("architectural_concepts_cleaned.csv")
write_label("development_concepts_cleaned.csv")
write_label("research_concepts_cleaned.csv")
