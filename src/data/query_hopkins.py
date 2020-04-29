
import requests


def query_hopkins (what) :
  url = 'https://api.covid19api.com/' + what
  payload = {}
  headers = {}
  response = requests.request("GET", url, headers=headers, data=payload)
  json_data = response.json()
  return json_data



