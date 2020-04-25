'''
This script fetches all data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login. Data fetched are form 2020-01-22T00:00:00Z to today.
'''
from absl import app
from absl import flags
import requests
from pymongo import MongoClient
import urllib.parse

import sys
sys.path.insert(0, "../")
from credentials import *

# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

flags.DEFINE_string("username", mongodb_username, "Username of MongoDB. Default is covics-19 user")
flags.DEFINE_string("password", mongodb_password, "Password of MongoDB. Default is ones's of covics-19 user")
flags.DEFINE_boolean("dry_run", True, "equivalent to `--no_request --no_db_write` (default is *TRUE*)")
flags.DEFINE_boolean("no_request", False, "Do not fetch data from Hopkins (for testing purpose)")
flags.DEFINE_boolean("no_db_write", False, "Do not modify the data base (for testing purpose)")

def main(argv):
    username = urllib.parse.quote_plus(FLAGS.username)
    password = urllib.parse.quote_plus(FLAGS.password)
    do_no_request = FLAGS.no_request
    do_no_db_write = FLAGS.no_db_write
    if (FLAGS.dry_run) :
      do_no_request = True
      do_no_db_write = True

    #----------------- Fetching data using REST call -----------------#
    url = "https://api.covid19api.com/all"
    payload = {}
    headers = {}
    print('Getting data from Hopkins DB...')
    if (not do_no_request) :
      response = requests.request("GET", url, headers=headers, data=payload)
      print('Data was retrieved.')
      json_data = response.json()
    else :
      print ('(dry_run/no_request: no request sent)')
      json_data = None

    # ----------------- Saving data in MongoDB -----------------#
    print('Connecting to MongoAtlas...')
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    print('Conected to MongoAtlas.')
    db = client['covics-19'] # get covid-19 DB
    hopkins = db['hopkins']
    print('Loading Hopkins data in MongoDB...')
    if (not do_no_db_write) :
      hopkins.insert_many(json_data)
    else :
      print ('(dry_run/no_db_write: no data inserted)')
    print('Hopkins data were loaded in MongoDB.')

if __name__ == "__main__":
    app.run(main)
