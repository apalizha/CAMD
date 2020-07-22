#  Copyright (c) 2019 Toyota Research Institute
"""
This module implements agent-tools for meta-analysis of
other agents
"""


import numpy as np
from taburu.table import ParameterTable
import pandas as pd

from camd.agent.base import HypothesisAgent


REGRESSOR_PARAMS = [
        {
            "@class": ["sklearn.linear_model.LinearRegression"],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        {
            "@class": ["sklearn.ensemble.RandomForestRegressor"],
            "n_estimators": [100],
            "max_features": list(np.arange(0.05, 1.01, 0.05)),
            "min_samples_split": list(range(2, 21)),
            "min_samples_leaf": list(range(1, 21)),
            "bootstrap": [True, False]
        },
        {
            "@class": ["sklearn.neural_network.MLPRegressor"],
            "hidden_layer_sizes": [
                # I think there's a better way to support this, but need to think
                (80, 50),
                (84, 55),
                (87, 60),
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        },
    ]


AGENT_PARAMS = [
    {
        "@class": ["camd.agent.stability.QBCStabilityAgent"],
        "n_query": [4, 6, 8],
        "n_members": list(range(2, 5)),
        "hull_distance": list(np.arange(0.05, 0.21, 0.05)),
        "training_fraction": [0.4, 0.5, 0.6],
        "model": REGRESSOR_PARAMS
    },
    {
        "@class": ["camd.agent.agents.AgentStabilityML5"],
        "n_query": [4, 6, 8],
        "hull_distance": [0.05, 0.1, 0.15, 0.2],
        "exploit_fraction": [0.4, 0.5, 0.6],
        "model": REGRESSOR_PARAMS
    },
]


def convert_parameter_table_to_dataframe(parameter_table, fillna=np.nan):
    """
    Converts parameter table in its current state to dataframe

    Args:
        parameter_table (ParameterTable): parameter table to
            convert to array
        fillna (int): a fill value for any remaining entries
            in the table

    Returns:
        (DataFrame): dataframe corresponding to parameter table
            data

    """
    df = pd.DataFrame(parameter_table, dtype="int64")
    df.index = ['-'.join([str(i) for i in row]) for row in np.array(df)]
    df['agent'] = [parameter_table.hydrate_index(i, construct_object=True)
                   for i in range(len(parameter_table))]
    df.fillna(fillna)
    return df


if __name__ == "__main__":
    first = ParameterTable(AGENT_PARAMS)
    first.hydrate_index(3)
    first.hydrate_index(3, construct_object=True)


class RegressorAgent(HypothesisAgent):
    """
    The agent used in meta_agent_campaign that selects the best campaign agent 
    hyperparameters for the next iteration. 
    """
    def __init__(
        self, model, target_prop, feature_cols, n_query=1, minimize=True
    ):
        """
        Args:
            model                   The ML model used to learn and predict the performance 
                                    of different campaign agents. 
            target_prop (str)       The target property that is being predicted. 
            feature_cols (list)     A list (of string) that represent the name of the feature
                                    data used in ML.
            n_query (int)           The number of queries allowed. Defaults to 1.
            minimize (bool)         If True, the agent with the minimum predicted target property 
                                    is selected. Else, the agent with the maximum predicted target 
                                    property is selected. 
        """
        self.model = model
        self.target_prop = target_prop
        self.feature_cols = feature_cols
        self.n_query = n_query
        self.minimize = minimize
        super(RegressorAgent).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
    
        feature_columns = self.feature_cols or candidate_data.columns.remove(self.target_prop)
        X_seed = seed_data[feature_columns]
        y_seed = seed_data[self.target_prop]

        self.model.fit(X_seed, y_seed)
        y_pred = self.model.predict(self.candidate_data[feature_columns])
        if self.minimize:
            selected = np.argsort(y_pred)[:self.n_query]
        else: 
            selected = np.argsort(-1 * y_pred)[:self.n_query]
        return candidate_data.iloc[selected]
