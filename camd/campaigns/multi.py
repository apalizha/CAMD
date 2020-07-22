"""
Prototype multi-fidelity campaign
"""
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

# Set pandas view options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from monty.os import cd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from camd.agent.base import HypothesisAgent, RandomAgent
from camd.analysis import AnalyzerBase
from camd.experiment.base import ATFSampler
from camd.campaigns.base import Campaign
from camd import CAMD_CACHE


class GenericMultiAgent(HypothesisAgent):
    """
    A multi-fidelity agent that takes in sklearn supervised regressor
    (GPR excluded) and generate hypotheses.
    """
    def __init__(self, target_prop='bandgap', target_prop_val=1.8,candidate_data=None, seed_data=None, 
                 preprocessor=StandardScaler(), model=None, 
                 query_criteria='cost_ratio', total_budget=None, exp_query_frac=0.2, theor_per_expt=2):
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.preprocessor = preprocessor
        self.model = model
        self.query_criteria = query_criteria
        self.total_budget = total_budget
        self.exp_query_frac = exp_query_frac
        self.theor_per_expt = theor_per_expt
        super(GenericMultiAgent).__init__()
        """
        Args:
            target_prop (str)        The property agent is trying to predict on,
                                     given feature space. i.e. bandgap.
            target_prop_val (float)  The ideal value of the target property.
            candidate_data (df)      A pd.DataFrame of candidate search space learning.
            seed_data (df)           A pd.DataFrame of training data for learning.
            preprocessor             A sklearn preprocessor that preprocess the features.
                                     It can be a single step or a pipeline. The default is
                                     StandardScaler().
            model                    The sklearn supervised machine learning regressor
                                     (GPR excluded).
            query_criteria (str)     Can be either 'cost_ratio' or 'number' to indicate which types of method
                                     will be used to query hypotheses. .                        
            total_budget (int)       The budget for the hypotheses query. If query_criteria is 'cost_ratio', this should 
                                     be an amount indicate costs, if query_criteria is 'number', this should be a 
                                     a total number of queries.
            exp_query_frac (float)   The fraction of the total budget that are used for experimental
                                     hypotheses queries. The value should be between 0 and 1.
            theor_per_expt (int)     Theory hypotheses budget per experimental candidates. In other
                                     words, the number of theory candidate selected to support each
                                     experimental candidates that predicted to be good, but the agent
                                     does not want to generate hypotheses yet. 
        """

    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.

        Args
            df              A pd.DataFrame where the features are extracted.
            add_fea(list)   Name of the additional features (str) used in ML.

        Returns
            feature_df      A pd.DataFrame that only contains the features used in ML.
                            By default, this will be compositional features.
        """
        magpie_columns = [column for column in df if column.startswith("MagpieData")]
        all_features = magpie_columns + add_fea
        feature_df = df[all_features]
        return feature_df

    def _process_data(self, candidate_data, seed_data):
        """
        Helper function that process data for ML model and returns
        np.ndarray of training features, training labels,
        test features and test labels.
        """
        X_train = self._get_features_from_df(seed_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_train = np.array(seed_data[self.target_prop])
        X_test = self._get_features_from_df(candidate_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_test = np.array(candidate_data[self.target_prop])
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test

    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between
        a composition and the seed data compositions. The similarity
        is reprsented by l2_norm.

        Args:
            comp(pd.core.series):    A specific composition represented by Magpie.
            seed_comps (df):         Compostions in seed represented by Magpie.
        """
        # match dimension to element wise operations
        comp = pd.DataFrame([comp]*len(seed_comps))
        l2_norm = np.linalg.norm(comp.values - seed_comps.values,  axis=1)
        return l2_norm

    def _query_hypotheses(self, candidate_data, seed_data):
        exp_candidates = candidate_data.loc[candidate_data.expt_data == 1]
        theor_candidates = candidate_data.loc[candidate_data.theory_data == 1]
        seed_data_fea = self._get_features_from_df(seed_data)
        exp_cands_fea = self._get_features_from_df(exp_candidates)

        # set up an empty df for queries
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
        
        # if there are no experimental data left in the candidate space, end the campaign
        if len(exp_candidates) == 0:
            return None
         
        # otherwise start the query   
        elif (len(exp_candidates) != 0) & (len(theor_candidates) == 0):
            for idx, exp in exp_candidates.iterrows(): 
                if self.query_criteria == 'number':
                    total_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    total_cost = np.sum(selected_hypotheses[self.query_criteria])

                if total_cost < self.total_budget:
                    selected_hypotheses = selected_hypotheses.append(exp)

        else:
            # first select experimental hypotheses
            exp_budget = self.total_budget * self.exp_query_frac
            threshold = 200 # TODO, make threshold params
            for idx, cand_fea in exp_cands_fea.iterrows(): 
                if self.query_criteria == 'number':
                    expt_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    expt_cost = np.sum(selected_hypotheses[self.query_criteria])
                
                if expt_cost < exp_budget:
                    normdiff_lst = self._calculate_similarity(cand_fea, seed_data_fea)
                    mask = (normdiff_lst <= threshold)
                    if len(normdiff_lst[mask]) > 1:
                        selected_hypotheses = selected_hypotheses.append(exp_candidates.loc[idx])

            # for remaining of the budget, select DFT hypotheses
            remained_exp_cands_fea = exp_cands_fea.drop(selected_hypotheses.index)
            theor_candidates_copy = theor_candidates.copy()
            for idx, cand_fea in remained_exp_cands_fea.iterrows():
                if self.query_criteria == 'number':
                    total_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    total_cost = np.sum(selected_hypotheses[self.query_criteria])
                    
                if (total_cost < self.total_budget) & (len(theor_candidates_copy) !=0):
                    theor_cands_fea = self._get_features_from_df(theor_candidates_copy)
                    theor_candidates_copy['normdiff'] = self._calculate_similarity(cand_fea, theor_cands_fea) 
                    theor_candidates_copy = theor_candidates_copy.sort_values('normdiff')
                    selected_hypotheses = selected_hypotheses.append(theor_candidates_copy.head(self.theor_per_expt))
                    theor_candidates_copy = theor_candidates_copy.drop(theor_candidates_copy.head(self.theor_per_expt).index)

        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        X_train, y_train, X_test, y_test = self._process_data(candidate_data, seed_data)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        candidate_data['prediction'] = y_pred
        candidate_data['dist_to_ideal'] = np.abs(self.target_prop_val - candidate_data['prediction'])
        candidate_data = candidate_data.sort_values(by=['dist_to_ideal'])
        hypotheses = self._query_hypotheses(candidate_data, seed_data)
        return hypotheses


class GPMultiAgent(HypothesisAgent):
    """
    Similar to Generic_MultiAgent, but the ML model is Gaussian Process regressor
    from GPy). This agent accounts for uncertainty when generating hypotheses.
    """
    def __init__(self, target_prop='bandgap', target_prop_val=1.8,
                 candidate_data=None, seed_data=None, preprocessor=StandardScaler(),
                 alpha=1.0, unc_percentile = 20, query_criteria='cost_ratio', total_budget=None):
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.unc_percentile = unc_percentile
        self.query_criteria = query_criteria
        self.total_budget = total_budget
        super(GPMultiAgent).__init__()

        """
        Args:
            target_prop (str)        The property agent is trying to predict on,
                                     given feature space. i.e. bandgap.
            target_prop_val (float)  The ideal value of the target property.
            candidate_data (df)      A pd.DataFrame of candidate search space learning.
            seed_data (df)           A pd.DataFrame of training data for learning.
            preprocessor             A sklearn preprocessor that preprocess the features.
                                     It can be a single step or a pipeline. The default is
                                     StandardScaler().
            alpha (float)            mixing parameter for uncertainties in UCB. It controls the 
                                     trade-off between exploration and exploitation. Defaults to 1.0. 
            unc_percentile (int)     A number between 0 and 100, and used to calculate an uncertainty threshold
                                     at that percentile value. The threshold is then used to decide if the
                                     agent is quering theory or experimental hypotheses. 
            query_criteria (str)     Can be either 'cost_ratio' or 'number' to indicate which types of method
                                     will be used to query hypotheses. .                        
            total_budget (int)       The budget for the hypotheses query. If query_criteria is 'cost_ratio', this should 
                                     be an amount indicate costs, if query_criteria is 'number', this should be a 
                                     a total number of queries. 
            """
    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.

        Args:
            df              A pd.DataFrame where the features are extracted.
            add_fea(list)   Name of the additional features (str) used in ML.

        Returns:
            feature_df      A pd.DataFrame that only contains the features used in ML.
                            by default, this will be compositional features.
        """
        magpie_columns = [column for column in df if column.startswith("MagpieData")]
        all_features = magpie_columns + add_fea
        feature_df = df[all_features]
        return feature_df

    def _process_data(self,  candidate_data, seed_data):
        """
        Helper function that process data for ML model and returns
        np.ndarray of training features, training labels,
        test features and test labels.
        """
        X_train = self._get_features_from_df(seed_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_train = np.array(seed_data[[self.target_prop]])
        X_test = self._get_features_from_df(candidate_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_test = np.array(candidate_data[[self.target_prop]])
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test
    
    def _query_hypotheses(self, candidate_data, seed_data):
        """
        idea behind MF-GP-UCB algorithm
        """
        seed_expt_data_chemsys = seed_data.loc[seed_data.expt_data==1].reduced_formula.values

        # get the ucb boundary, which is min(pred_ucb) of various fidelities for given chemsys
        candidate_data = candidate_data.sort_values(by=['reduced_formula', 'pred_unc'])
        ucb_boundary = candidate_data.drop_duplicates(subset='reduced_formula', keep='first')
        ucb_boundary = ucb_boundary.sort_values('dist_to_ideal')
        unc_thres = np.percentile(np.array(ucb_boundary.pred_unc), self.unc_percentile)
        
        # query hypotheses
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns) 
        for formula in ucb_boundary.reduced_formula.values:
            if self.query_criteria == 'number':
                total_cost = len(selected_hypotheses)
            elif self.query_criteria == 'cost_ratio':
                total_cost = np.sum(selected_hypotheses[self.query_criteria])
               
            if total_cost < self.total_budget:
                # only query the composition if its expt data does not already exist in seed
                if formula not in seed_expt_data_chemsys:
                    theory = candidate_data.loc[(candidate_data.reduced_formula == formula)&
                                                (candidate_data.theory_data == 1)]
                    if len(theory) !=0 and theory.pred_unc.values[0] >= unc_thres:
                        selected_hypotheses = selected_hypotheses.append(theory)
                    else: 
                        expt = candidate_data.loc[(candidate_data.reduced_formula == formula)&
                                          (candidate_data.expt_data == 1)]
                        selected_hypotheses = selected_hypotheses.append(expt) 
        return selected_hypotheses     

    def get_hypotheses(self, candidate_data, seed_data):
        X_train, y_train, X_test, y_test = self._process_data(candidate_data, seed_data)
        gp = GPy.models.GPRegression(X_train, y_train)
        gp.optimize('bfgs', max_iters=200)
        y_pred, var = gp.predict(X_test)
        pred_ucb = y_pred + self.alpha * var**0.5
        dist_to_ideal = np.abs(self.target_prop_val - pred_ucb)
        candidate_data['pred_unc'] = self.alpha * var**0.5
        candidate_data['dist_to_ideal'] = dist_to_ideal 
        hypotheses = self._query_hypotheses(candidate_data, seed_data)
        return hypotheses


class MultiAnalyzer(AnalyzerBase):
    """
    The multi-fidelity analyzer.
    """
    def __init__(self, target_prop, prop_range, total_expt_queried=0, 
                 total_expt_discovery=0, total_cost=0.0, extra_stats=None):
        """
        Args:
            target_prop (str)        The name of the target property, e.g. "bandgap".
            prop_range (list)        The range of the target property that is considered
                                     ideal.
            total_expt_queried (int) The total experimental queries after nth iteration. 
            tot_expt_discovery (int) The total experimental discovery after nth iteration.                     
            total_cost(float)        The total cost of the hypotheses after nth iteration.       
            extra_stats (dict)       A dictionary with key/value pairings that correspond
                                     to additional statistics you want to fetch. Only the
                                     recorded ones (TODO: x,y,z) can be fetched.
        """
        self.target_prop = target_prop
        self.prop_range = prop_range
        self.total_expt_queried = total_expt_queried
        self.total_expt_discovery = total_expt_discovery
        self.total_cost = total_cost
        self.extra_stats = extra_stats # TODO

    def _filter_df_by_prop_range(self, df):
        """
        Helper function that filters df by property range

        Args:
            df   A pd.Dataframe to be filtered.
        """
        return df.loc[(df[self.target_prop] >= self.prop_range[0]) &
                      (df[self.target_prop] <= self.prop_range[1])]

    def analyze(self, new_experimental_results, seed_data):  #, agent=None):
        positive_hits = self._filter_df_by_prop_range(new_experimental_results)

        new_expt_hypotheses = new_experimental_results.loc[new_experimental_results['expt_data'] == 1]
        new_discoveries = self._filter_df_by_prop_range(new_expt_hypotheses)
        iter_cost = np.sum(new_experimental_results['cost_ratio'])
        # Uncertainty?

        # total discovery = up to (& including) the current iteration
        new_seed = seed_data.append(new_experimental_results)
        self.total_expt_queried += len(new_expt_hypotheses)
        self.total_expt_discovery += len(new_discoveries)
        self.total_cost += iter_cost
        
        summary = pd.DataFrame(
                {
                 "expt_queried": [len(new_expt_hypotheses)],
                 "total_expt_queried": [self.total_expt_queried], 
                 "new_expt_discovery": [len(new_discoveries)],
                 "iteration_cost": [iter_cost],
                 "total_expt_discovery": [self.total_expt_discovery],
                 "total_cost": [self.total_cost], 
                 "total_regret": [self.total_expt_queried - self.total_expt_discovery], 
                 "success_rate": [self.total_expt_discovery/self.total_expt_queried]
            }
        )
        return summary, new_seed
