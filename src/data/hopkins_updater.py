"""
example of query for the update:
https://api.covid19api.com/country/south-africa/status/confirmed?from=2020-03-01T00:00:00Z&to=2020-04-01T00:00:00Z

Ive added a collection 'checkpoint' to get the date of the last update
check for document with content such as:
last_hopkins_update:"2020-04-25"
TODO: thats not so great in terms of synchronization (somebody competent should look into this please)
"""
import datetime
import sys

sys.path.insert(0, "../")
from credentials import *
from utils . mongodb_utils import update_checkpoint, get_checkpoint, get_mongodb_collection

from query_hopkins import query_hopkins



def fetch_new_data_from_hopkins (quiet = False, dryrun = False) :
  today = datetime . datetime . utcnow () . date ()
  date_last_update = datetime . date . fromisoformat (get_checkpoint ())
  if (today <= date_last_update) :
    if (not quiet) :
      print ('Warning: Hopkins data has already been updated.')
    return 
  start_date = (date_last_update + datetime . timedelta (days = 1)) . isoformat ()
  end_date = (today + datetime . timedelta (days = 1)) . isoformat ()
  # this doesnt work like that unfortunately:
  hopkins_api_query = 'all?from=' + start_date + 'T00:00:00Z&to=' + end_date + 'T00:00:00Z'
  if (not dryrun) :
    json_data = query_hopkins (hopkins_api_query)
  else :
    print (f'Query={hopkins_api_query} (not submitted)')
    json_data = None
  if (not dryrun and (json_data is None)) :
    return
  hopkins_collection = get_mongodb_collection ('covics-19', collection_name = 'hopkins')
  if (not dryrun) :
    hopkins_collection . insert_many (json_data)
    # TODO: probably not entirely safe but im assuming that this script wont be called more than once a day so should be fine
    update_checkpoint (date = today . isoformat ())
  else :
    print ('Database not updated (dryrun)')

if __name__ == "__main__":
  fetch_new_data_from_hopkins ()
