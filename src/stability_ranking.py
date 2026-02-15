import pandas as pd


def rank_operators(df, error_column="pu_error"):
    """
    Rank operators by mean perceptual error.
    Lower error = more stable / better.
    """

    if "operator" not in df.columns:
        raise ValueError("DataFrame must contain 'operator' column.")

    ranking = (
        df.groupby("operator")[error_column]
        .mean()
        .sort_values()
        .reset_index()
    )

    ranking.columns = ["operator", f"mean_{error_column}"]

    return ranking