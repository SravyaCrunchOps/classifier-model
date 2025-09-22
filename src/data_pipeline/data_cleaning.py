

def data_cleaning(df):
    null_summary = df.isnull().sum()
    null_columns = null_summary[null_summary > 0]
    # print(null_columns)

    return null_columns
