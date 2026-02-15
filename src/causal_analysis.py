def highlight_causality(df):
    return df["highlight_ratio"].corr(df["pu_error"])