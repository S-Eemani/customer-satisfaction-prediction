import logging
from zenml import step
import pandas as pd
@step
def test_model(df:pd.DataFrame) -> None:
    """evaluates the model on the given df

    Args:
        df (pd.DataFrame): the ingested data
    """
    None