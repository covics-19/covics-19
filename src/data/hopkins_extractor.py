'''
This script fetches data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login
'''
from absl import app
from absl import flags
import http.client
import mimetypes
from pymongo import MongoClient
from pprint import pprint
import urllib.parse
import json


# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

flags.DEFINE_string("username", "covics-19", "Username of MongoDB. Default is covics-19 user")
flags.DEFINE_string("password", "Coron@V!rus2020", "Password of MongoDB. Default is ones's of covics-19 user")

def main(argv):
    username = urllib.parse.quote_plus(FLAGS.username)
    password = urllib.parse.quote_plus(FLAGS.password)

    #----------------- Fetching data using REST call -----------------#
    conn = http.client.HTTPSConnection("api.covid19api.com")
    payload = ''
    headers = {}
    conn.request("GET", "/all", payload, headers)
    print('Getting data from Hopkins DB...')
    res = conn.getresponse()
    data = res.read()
    print('Data was retrieved.')
    json_data = json.loads(data.decode("utf-8"))

    # ----------------- Saving data in MongoDB -----------------#
    print('Connecting to MongoAtlas...')
    client = MongoClient("mongodb+srv://" + username + ":" + password + "@cluster0-pjnfk.mongodb.net/test?retryWrites=true&w=majority")
    print('Conected to MongoAtlas.')
    db = client['covics-19'] # get covid-19 DB
    hopkins = db['hopkins']
    print('Loading Hopkins data in MongoDB...')
    hopkins.insert_many(json_data)
    print('Hopkins data were loaded in MongoDB.')


    '''
    # TODO: temporary json file
    # Saving as JSON
    import json
    json_data = json.loads(data.decode("utf-8"))
    with open('hopkins.json', 'w') as outfile:
        json.dump(json_data, outfile)'''

if __name__ == "__main__":
    app.run(main)
    # TODO: split in function to extract all data once and a fanction to extract only data of last day