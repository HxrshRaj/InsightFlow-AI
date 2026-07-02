import pandas as pd

from backend.cleaning import clean_dataset


def test_clean_dataset_returns_cleaned_frame_and_summary() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "age": [25, None, 30, 40],
            "city": ["NY", None, "NY", "LA"],
            "score": [1.0, 2.0, 2.0, 3.0],
        }
    )
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)

    cleaned_df, summary, log = clean_dataset(df)

    assert cleaned_df is not None
    assert summary["rows_after"] <= summary["rows_before"]
    assert summary["duplicates_removed"] >= 1
    assert summary["missing_values_fixed"] >= 1
    assert isinstance(log, list)
    assert len(log) >= 1
    assert len(df) == 5
    assert df.iloc[0].to_dict() == {"id": 1, "age": 25.0, "city": "NY", "score": 1.0}
    assert pd.isna(df.iloc[1]["age"])
    assert pd.isna(df.iloc[1]["city"])
