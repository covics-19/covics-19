
from credentials import *

from pymongo import MongoClient
import urllib.parse

def get_mongodb_collection (data_base_name, collection_name = None) :
  username = urllib.parse.quote_plus(mongodb_username)
  password = urllib.parse.quote_plus(mongodb_password)
  client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
  if (collection_name is None) :
    return client [data_base_name]
  return client [data_base_name] [collection_name]

