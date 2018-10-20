import os
import sys
import json
import requests
import socket
import time
from requests.auth import HTTPBasicAuth

GATEWAY_URL = "https://gateway-staging.ncrcloud.com"

# authentication used for NCR API
auth = HTTPBasicAuth('acct:bizniz@biznizserviceuser', 'hackgsuwinner')
headers = {
    'nep-application-key': '8a0084a165d712fd016646ec1613003c',
    'nep-organization': 'ncr-market',
    'nep-service-version': '2.2.1:2'
}
catalog = None

def get_catalog():
    # Connecting to the API and returning the entire catalog avaiable
    r = requests.get(GATEWAY_URL + "/catalog/items/snapshot", auth=auth, headers=headers)
    print(r.content)
    if r.status_code > 400:
        return None
    else:
        global catalog
        catalog = json.loads(r.content)

get_catalog()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 10001))
sock.listen(5)
while True:
    conn, addr = sock.accept()
    while True:
        for snap in catalog['snapshot']:
            resp = json.dumps({"name": snap['shortDescription']['values'][0]['value'], "itemId": snap['itemId']})
            print(resp)
            conn.sendall(resp)
            time.sleep(2.0)