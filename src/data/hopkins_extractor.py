'''
This script fetches all data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login. Data fetched are form 2020-01-22T00:00:00Z to today.
'''
from absl import app
from absl import flags
import requests
from pymongo import MongoClient
import urllib.parse

# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

flags.DEFINE_string("username", "covics-19", "Username of MongoDB. Default is covics-19 user")
flags.DEFINE_string("password", "Coron@V!rus2020", "Password of MongoDB. Default is ones's of covics-19 user")

def main(argv):
    username = urllib.parse.quote_plus(FLAGS.username)
    password = urllib.parse.quote_plus(FLAGS.password)

    #----------------- Fetching data using REST call -----------------#
    url = "https://api.covid19api.com/all"
    payload = {}
    headers = {}
    print('Getting data from Hopkins DB...')
    response = requests.request("GET", url, headers=headers, data=payload)
    print('Data was retrieved.')
    json_data = response.json()

    # ----------------- Saving data in MongoDB -----------------#
    print('Connecting to MongoAtlas...')
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    print('Conected to MongoAtlas.')
    db = client['covics-19'] # get covid-19 DB
    hopkins = db['hopkins']
    print('Loading Hopkins data in MongoDB...')
    hopkins.insert_many(json_data)
    print('Hopkins data were loaded in MongoDB.')

if __name__ == "__main__":
    app.run(main)