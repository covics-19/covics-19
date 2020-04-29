"""
the function load_model_data comes from dmpiergiacomo PR (#33)
the main.py uses load_data_for_model_growth
which one is the latest?
"""

import json
from bson import ObjectId
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../')
from utils.mongodb_utils import get_mongodb_collection

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def _fetch_hopkins_from_db():
    '''
    This script fetches latest Hopkins data from our MongoDB and returns them as list of dictionaries
    :return: List of dictionaries containing Hopkins data
    '''
    entries_list = []
    hopkins = get_mongodb_collection ('covics-19', collection_name = 'hopkins') . find ()
    for elem in hopkins:
        entries_list.append(elem)
    # entries_json = JSONEncoder().encode(entries_list) # use this to return in str format
    return entries_list



def load_model_data():
    '''
    This script fetches latest Hopkins data from our MongoDB to feed our prediction model
    :return:
    '''
    entries_list = _fetch_hopkins_from_db()
    df = pd.DataFrame(entries_list)  # DataFrame of all Hopkins cases
    # we don't groupby lat and lon ---> hopkins mismatches on lat and lon values are therefore avoided
    return df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'CountryCode', 'Date']) \
               ['Confirmed'].aggregate('first').unstack().reset_index(). \
               replace({'Province': {'to_be_removed': ''}}), \
           df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'CountryCode', 'Date']) \
               ['Deaths'].aggregate('first').unstack().reset_index(). \
               replace({'Province': {'to_be_removed': ''}}), \
           df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'CountryCode', 'Date']) \
               ['Recovered'].aggregate('first').unstack().reset_index(). \
               replace({'Province': {'to_be_removed': ''}})


               
def load_data_for_model_growth():
    entries_list = _fetch_hopkins_from_db()
    df = pd.DataFrame(entries_list)
    countrywide_df = df.groupby(['Country','CountryCode', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
    return countrywide_df

