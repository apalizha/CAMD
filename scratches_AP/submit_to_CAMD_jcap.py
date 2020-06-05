import os
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

from camd.agent.base import HypothesisAgent
from camd.analysis import AnalyzerBase
from camd.experiment.base import ATFSampler
from camd.campaigns.base import Campaign

path = 'CAMD/scratches/JCAP data/'
jcap_pickle = 'jcap_optical_encoding.pickle'

jcap_df = pd.read_pickle(jcap_pickle)
print('This dataset has {} samples, {} features'.format(jcap_df.shape[0], jcap_df.shape[1]))

print('Processing Data ....')
# process the compositions so they are not sparsed representation
# There are two columns about composition. One is the full composition in dictionary format (e.g. {'Bi': 0.95, 'Mn': 0.05}).
# The other is the element combination in tuples, sorted alphabetically.
compositions = []
for index, row in jcap_df.iterrows():
    composition = {elem: round(ratio,3) for elem, ratio in zip(row['Fe':'Rb'].index, row['Fe':'Rb']) if ratio != 0}
    compositions.append(composition)

processed_jcap_df = jcap_df[['bandgap']].copy()
processed_jcap_df['full_composition'] = compositions

elem_combo_sorted = [tuple(sorted(comp.keys())) for comp in compositions]
processed_jcap_df['elem_combo'] = elem_combo_sorted

# get the binary and ternary compound
selected_jcap_df = processed_jcap_df[(processed_jcap_df.elem_combo.map(len)==2)|(processed_jcap_df.elem_combo.map(len)==3)]

# Helper script taken from
# https://github.awsinternal.tri.global/materials/camd_api/blob/master/scripts/submit_LaFeAs_class.py

import argparse
from camd_apigw.s3_utils import submit_chemsys, CAMD_SYNC_S3_BUCKET, \
    CAMD_DEFAULT_CAMPAIGN
import itertools
from tqdm import tqdm


argparser = argparse.ArgumentParser()
argparser.add_argument("--campaign", "-r", default=CAMD_DEFAULT_CAMPAIGN,
                       help="Type of run, e. g. proto-dft or oqmd-atf")
argparser.add_argument("--bucket", "-b", default=CAMD_SYNC_S3_BUCKET,
                       help="Bucket for submission, e.g. camd-runs or camd-test")

def get_possible_chemsyses(combinations):
    """Function to get all of the oxide to test"""
    return ['-'.join(combination) +'-O' for combination in combinations]


print('Parsing and submitting data....')
# all the unique composition combination from JCAP data (sorted alphabetically and not duplicated)
elem_combo_set = set(list(selected_jcap_df['elem_combo']))
combinations_of_interest = sorted(list(elem_combo_set), reverse=True)

# CAMD is last in, first out, so we will sort the list above by length
sorted_interest = sorted(combinations_of_interest, key=len, reverse=True)[-10:]

args = argparser.parse_args()
for chemsys in tqdm(get_possible_chemsyses(sorted_interest)):
    submit_chemsys(chemsys, bucket_name=args.bucket,
                   campaign=args.campaign)
