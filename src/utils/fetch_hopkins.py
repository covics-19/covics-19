from pymongo import MongoClient
import urllib.parse
import json
from bson import ObjectId
import pandas as pd
import numpy as np


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
    username = urllib.parse.quote_plus("covics-19")
    password = urllib.parse.quote_plus("Coron@V!rus2020")
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    db = client['covics-19']
    hopkins = db["hopkins"].find()
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
    df = pd.DataFrame(entries_list)         # DataFrame of all Hopkins cases
    # we don't groupby lat and lon ---> hopkins mismatches on lat and lon values are therefore avoided
    return df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'Date'])['Confirmed'].\
               aggregate('first').unstack().reset_index().replace({'Province':{'to_be_removed': np.nan}}), \
           df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'Date'])['Deaths'].\
               aggregate('first').unstack().reset_index().replace({'Province':{'to_be_removed': np.nan}}), \
           df.reset_index().groupby([df['Province'].fillna('to_be_removed'), 'Country', 'Date'])['Recovered'].\
               aggregate('first').unstack().reset_index().replace({'Province':{'to_be_removed': np.nan}})

