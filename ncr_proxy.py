import os
import sys
import json
import requests
import socket
import numpy as np
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()
catalog = None
suggestions = None
GATEWAY_URL = os.getenv('NCR_GATEWAY_ENDPOINT')

payload_false = "{\"availableForSale\":false}"
payload_true = "{\"availableForSale\":true}"

endpoint_suggestions = "/catalog/items/suggestions?"
endpoint_available = "/ias/item-availability/"
endpoint_put = "/ias/item-availability/"
endpoint_get = "/catalog/items/snapshot/"
endpoint_snapshots = "/catalog/items/snapshot"

query = "descriptionPattern=corn+flakes"

# authentication used for NCR API
auth = HTTPBasicAuth('acct:bizniz@biznizserviceuser', 'hackgsuwinner')
headers = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'nep-application-key': '8a0084a165d712fd016646ec1613003c',
    'nep-organization': 'bizniz',
}

headersIAS = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'nep-application-key': '8a0084a165d712fd016646ec1613003c',
    'nep-organization': 'bizniz',
    'nep-enterprise-unit': '748c4a76cd714502b3bff0525cb48ca7'
}


def prep_data():
    r_get = requests.get(GATEWAY_URL + endpoint_get, auth=auth, headers=headers)
    data = json.loads(r_get.content)
    for item in data['snapshot']:
        product_id = item['itemId']['itemCode']
        if np.random.choice(2, 1) == 2:
            r_put = requests.put(GATEWAY_URL + endpoint_put + product_id, auth=auth, headers=headersIAS,
                                 data=payload_true)
        else:
            r_put = requests.put(GATEWAY_URL + endpoint_put + product_id, auth=auth, headers=headersIAS,
                                 data=payload_false)


def get_suggestions():
    # Connecting to the API and returning json based on query
    r = requests.get(GATEWAY_URL + endpoint_suggestions + query, auth=auth, headers=headers)
    print(r.content)
    if r.status_code > 400:
        return None
    else:
        global suggestions
        suggestions = json.loads(r.content)
    # sendData(suggestions)

    # testing availability with constrained items
    item = suggestions['pageContent'][0]
    itemId = item['itemId']
    print(itemId)

    status = is_available(itemId)

    print(status)


def get_catalog():
    # connecting to the API and returning json based on query
    r = requests.get(GATEWAY_URL + endpoint_snapshots, auth=auth, headers=headers)
    print(r.content)
    if r.status_code > 400:
        return None
    else:
        global catalog
        catalog = json.loads(r.content)
    # sendData(catalog)


def is_available(item_id):
    # request availability of this itemID
    product_id = item_id.get('itemCode')
    # connecting to the API and returning json based on query
    r = requests.get(GATEWAY_URL + endpoint_available + product_id, auth=auth, headers=headersIAS)
    print(r.content)
    if r.status_code > 400:
        return None
    else:
        isItThere = json.loads(r.content)

    if isItThere['availableForSale']:
        return True
    else:
        return False


def send_data(data):
    isSnapshot = 'snapshot' in data


if __name__ == '__main__':
    get_catalog()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 10001))
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        while True:
            for snap in catalog['snapshot']:
                availability = is_available(snap['itemId'])
                resp = json.dumps({"name": snap['shortDescription']['values'][0]['value'], "itemId": snap['itemId'],
                                   "is_available": availability})
                # print(resp)
                try:
                    conn.sendall(resp)
                    data = conn.recv(10)
                    print("Data from client: {}".format(data))
                except:
                    sys.exit(1)
