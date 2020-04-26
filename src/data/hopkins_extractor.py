'''
This script fetches all data from Hopkins DB at https://covid19api.com/#details and populate our MongoDB at
https://account.mongodb.com/account/login. Data fetched are form 2020-01-22T00:00:00Z to today.
'''
from absl import app
from absl import flags

import sys
sys.path.insert(0, "../")
from credentials import *
from utils . mongodb_utils import update_checkpoint, get_mongodb_collection
from query_hopkins import query_hopkins

# ------------------ Parameters ------------------- #
FLAGS = flags.FLAGS

flags.DEFINE_string("username", mongodb_username, "Username of MongoDB. Default is covics-19 user")
flags.DEFINE_string("password", mongodb_password, "Password of MongoDB. Default is ones's of covics-19 user")
flags.DEFINE_boolean("dry_run", True, "equivalent to `--no_request --no_db_write` (default is *TRUE*)")
flags.DEFINE_boolean("no_request", False, "Do not fetch data from Hopkins (for testing purpose)")
flags.DEFINE_boolean("no_db_write", False, "Do not modify the data base (for testing purpose)")

def main(argv):
    do_no_request = FLAGS.no_request
    do_no_db_write = FLAGS.no_db_write
    if (FLAGS.dry_run) :
      do_no_request = True
      do_no_db_write = True

    #----------------- Fetching data using REST call -----------------#
    print('Getting data from Hopkins DB...')
    if (not do_no_request) :
      json_data = query_hopkins ('all')
      print('Data was retrieved.')
    else :
      print ('(dry_run/no_request: no request sent)')
      json_data = None

    # ----------------- Saving data in MongoDB -----------------#
    print('Connecting to MongoAtlas...')
    hopkins = get_mongodb_collection ('covics-19', collection_name = 'hopkins')
    print('Conected to MongoAtlas.')
    print('Loading Hopkins data in MongoDB...')
    if (not do_no_db_write) :
      hopkins.insert_many(json_data)
      update_checkpoint ()
    else :
      print ('(dry_run/no_db_write: no data inserted)')
    print('Hopkins data were loaded in MongoDB.')

if __name__ == "__main__":
    app.run(main)
