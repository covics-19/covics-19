'''
This script fetches today's data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login.
'''
from absl import app
from absl import flags
import requests
from pymongo import MongoClient
import urllib.parse
from datetime import datetime

# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

flags.DEFINE_string("username", "covics-19", "Username of MongoDB. Default is covics-19 user")
flags.DEFINE_string("password", "Coron@V!rus2020", "Password of MongoDB. Default is ones's of covics-19 user")

def main(argv):
    username = urllib.parse.quote_plus(FLAGS.username)
    password = urllib.parse.quote_plus(FLAGS.password)

    # -------------- Fetching countries using REST call --------------#
    url = "https://api.covid19api.com/countries"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    countries = response.json()

    # ------------------------ Today timestamp -----------------------#
    now = datetime.now()
    yesterday = now.replace(day=(now.day -1), hour=23, minute=59, second=59, microsecond=0) # midnight time as hopkins convention
    yesterday = yesterday.strftime('%Y-%m-%dT%H:%M:%SZ')

    # --------------------- Connecting to MongoDB --------------------#
    print('Connecting to MongoAtlas...')
    client = MongoClient(
        "mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    print('Conected to MongoAtlas.')
    db = client['covics-19']  # get covid-19 DB
    hopkins = db['hopkins']

    #----------------- Fetching data using REST call -----------------#
    for country in countries:
        print(country)
        slug = country['Slug']
        url = "https://api.covid19api.com/live/country/" + slug + "/status/confirmed/date/" + str(yesterday)
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        json_response = response.json()
        for json_data in json_response:
            print(json_data)

            # ----------------- Saving data in MongoDB -----------------#
            print('Loading Hopkins data in MongoDB...')
            hopkins.insert_one(json_data)
            print('Hopkins data were loaded in MongoDB.')

if __name__ == "__main__":
    app.run(main)