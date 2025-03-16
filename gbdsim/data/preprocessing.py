import numpy as np
import pandas as pd


def clean_and_binarize_classification(df: pd.DataFrame) -> pd.DataFrame:
    target_variable = df.iloc[:, -1]
    df = df.loc[~target_variable.isna()]
    target_variable = target_variable[~target_variable.isna()]

    unique_categories = np.array(
        sorted(target_variable.drop_duplicates().tolist())
    )
    one_category_subset = np.random.uniform(size=len(unique_categories)) <= 0.5
    if one_category_subset.all() or (~one_category_subset).all():
        one_category_subset[-1] = not one_category_subset[-1]
    one_category_subset = unique_categories[one_category_subset]
    target_variable = target_variable.isin(one_category_subset).astype(int)
    target_name = df.columns[-1]
    df[target_name] = target_variable
    return df


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()[:-1]
    id_columns = filter(
        lambda c: c.lower().startswith("id") or c.lower().endswith("_id"),
        columns,
    )
    df = df.drop(columns=id_columns)
    df = df.select_dtypes(exclude=pd.Timestamp)
    return df
