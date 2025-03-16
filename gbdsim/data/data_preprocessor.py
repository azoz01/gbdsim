from typing import Tuple

import pandas as pd
from dataset2vec.utils import DataUtils
from torch import Tensor

from gbdsim.data.preprocessing import (
    clean_and_binarize_classification,
    remove_unwanted_columns,
)


class DataPreprocessor:

    def preprocess_pandas_data(
        self, df: pd.DataFrame
    ) -> Tuple[Tensor, Tensor]:
        df = clean_and_binarize_classification(df)
        df = remove_unwanted_columns(df)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        return Tensor(
            DataUtils.get_preprocessing_pipeline().fit_transform(X).values  # type: ignore # noqa: E501
        ), Tensor(y.values)
