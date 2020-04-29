'''
This script fetches today's data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login.
'''
from absl import app
from datetime import datetime
from absl import flags
import requests
import sys

sys . path . insert (0, "../")
import credentials
from utils . mongodb_utils import update_checkpoint, get_checkpoint, get_mongodb_collection

# ------------------ Parameters ------------------- #
FLAGS = flags . FLAGS

flags . DEFINE_string ('username', credentials . mongodb_username, 'Username of MongoDB. Default is covics-19 user')
flags . DEFINE_string ('password', credentials . mongodb_password, 'Password of MongoDB. Default is ones\'s of covics-19 user')
flags . DEFINE_boolean ('dryrun', False, '')

def main(argv):

    # ------------------------ Today timestamp -----------------------#
    now = datetime.now()
    yesterday = now.replace(day=(now.day -1), hour=23, minute=59, second=59, microsecond=0) # midnight time as hopkins convention
    yesterday = yesterday.strftime('%Y-%m-%dT%H:%M:%SZ')

    # --------------------- Connecting to MongoDB --------------------#
    print ('Connecting to MongoAtlas...')
    hopkins = get_mongodb_collection ('covics-19',
                                      collection_name = 'hopkins',
                                      mongodb_username = FLAGS . username,
                                      mongodb_password = FLAGS . password)
    print ('Conected to MongoAtlas.')

    # -------------- Fetching countries using REST call --------------#
    url = "https://api.covid19api.com/countries"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    countries = response.json()
    
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
            if (FLAGS . dryrun) :
              print ('(dry run: nothing updated)')
            else :
              hopkins.insert_one(json_data)
            print('Hopkins data were loaded in MongoDB.')

if __name__ == "__main__":
    app.run(main)
