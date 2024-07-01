import pandas as pd
from IPython import display

def df_overview(df: pd.DataFrame):
    """
    This function takes a pandas dataframe as input and returns the shape, head, tail, and info of the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
            
    Returns:
    shape, head, tail, and info of the dataframe.
    """
    display("-------Data Preview-------")
    display(f"Shape: {df.shape}")
    display(f"Head and tail preview:")
    display(df)
    display(f"Df info:")
    if df.info(verbose=True) != None:
        display(df.info(verbose=True))

def univariate_preview(df: pd.DataFrame, cols: list, describe: bool =True):
    """
    This function takes a pandas dataframe and a list of columns as input and returns the shape, head, tail, info, value counts, and summary statistics of the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    cols (list): The list of columns to be analyzed.
    describe (bool): If True, display the summary statistics of the dataframe.

    Returns:
    shape, head, tail, info, value counts, and summary statistics of the dataframe.
    """
    display("-------Data Preview-------")
    display(df[cols].head())
    
    display("-------Value Counts-------")
    list = []
    for col in cols:
        list.append(
            [col,
            df[col].dtypes,
            df[col].nunique(),
            df[col].value_counts().iloc[:5].index.tolist(),
            "{:.2f}%".format(df[col].isna().mean()*100)]
            )
    display(pd.DataFrame(list, 
                         columns = ['columns', 'dtypes', 'nunique', 'top5', 'na%']
                         ).sort_values('nunique', ascending=False))
    
    if describe:
        display("-------Summary Stats-------")
        display(pd.concat([
            df[cols].describe(),
            df[cols].skew().to_frame('skewness').T,
            df[cols].kurtosis().to_frame('kurtosis').T,
        ]))

