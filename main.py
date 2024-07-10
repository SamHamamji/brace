import os
from pathlib import Path
import typing

import pandas as pd

from utils import BraceDataset


def get_dataframe(
    annotations_dir: str, dataset_type: typing.Literal["train"] | typing.Literal["test"]
) -> pd.DataFrame:
    raw_df = pd.read_csv(os.path.join(annotations_dir, "sequences.csv"))

    train_df = pd.read_csv(
        os.path.join(annotations_dir, f"sequences_{dataset_type}.csv")
    )

    return raw_df[raw_df.uid.isin(train_df.uid)]


if __name__ == "__main__":
    sequences_path = Path("./data")
    train_df = get_dataframe("./annotations", "train")
    test_df = get_dataframe("./annotations", "test")

    brace_train = BraceDataset(sequences_path, train_df)
    print(f"Loaded BRACE training set! We got {len(brace_train)} training sequences")
    skeletons_train, metadata_train = brace_train[0]
    print(metadata_train)

    brace_test = BraceDataset(sequences_path, test_df)
    print(f"Loaded BRACE test set! We got {len(brace_test)} testing sequences")
    skeletons_test, metadata_test = brace_test[0]
    print(metadata_test)
