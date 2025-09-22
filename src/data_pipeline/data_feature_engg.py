
def feature_engg(df):
    df['can_fly'] = (df['airborne'] == 1) & (df['feathers'] == 1)
    df['can_swim'] = (df['aquatic'] == 1) & (df['fins'] == 1)
    df['is_domestic_pet'] = (df['domestic'] == 1) & (df['catsize'] == 1)

    # convert False, True to numerical 0, 1
    df['can_fly'] = df['can_fly'].astype(int)
    df['can_swim'] = df['can_swim'].astype(int)
    df['is_domestic_pet'] = df['is_domestic_pet'].astype(int)

    return df
