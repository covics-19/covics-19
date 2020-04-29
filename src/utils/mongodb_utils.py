

import sys
sys.path.insert(0, "../")
import credentials

import datetime
from pymongo import MongoClient
import urllib.parse

def get_mongodb_collection (data_base_name,
                            collection_name = None,
                            mongodb_username = credentials . mongodb_username,
                            mongodb_password = credentials . mongodb_password) :
  username = urllib.parse.quote_plus(mongodb_username)
  password = urllib.parse.quote_plus(mongodb_password)
  client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
  if (collection_name is None) :
    return client [data_base_name]
  return client [data_base_name] [collection_name]


def update_checkpoint (date = None) :
  # refresh the entire collection with only one document
  if (date is None) :
    # TODO: there *must* be a simpler way to get today's date in iso format
    date = datetime . datetime . utcnow () . date () . isoformat ()
  checkpoint_collection = get_mongodb_collection ('covics-19', collection_name = 'checkpoint')
  checkpoint_collection . delete_many ({})
  checkpoint_collection . insert_one ({ 'last_hopkins_update' : date })

def get_checkpoint () :
  checkpoint = get_mongodb_collection ('covics-19', collection_name = 'checkpoint')  . find () [0]
  return checkpoint ['last_hopkins_update']


