import os
import sys
import json
import requests
import socket
import time
import random
import numpy as np
from requests.auth import HTTPBasicAuth

GATEWAY_URL = "https://gateway-staging.ncrcloud.com"

# authentication used for NCR API
auth = HTTPBasicAuth('acct:bizniz@biznizserviceuser', 'hackgsuwinner')
headers = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'nep-application-key': '8a0084a165d712fd016646ec1613003c',
    'nep-organization': 'bizniz',
    'nep-enterprise-unit': '748c4a76cd714502b3bff0525cb48ca7',
    'nep-service-version': '2.2.1:2'
}

headersIAS = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'nep-application-key': '8a0084a165d712fd016646ec1613003c',
    'nep-organization': 'bizniz',
    'nep-enterprise-unit': '748c4a76cd714502b3bff0525cb48ca7'
}

catalog = None
suggestions = None

def prepData():
    data = None
    endpoint_put = "/ias/item-availability/"
    endpoint_get = "/catalog/items/snapshot/"
    payload_false = "{\"availableForSale\":false}"
    payload_true = "{\"availableForSale\":true}"
    
    r_get = requests.get(GATEWAY_URL + endpoint_get, auth=auth, headers=headers)

    data = json.loads(r_get.content)
    
    for item in data['snapshot']:
        product_id = item['itemId']['itemCode']
        if np.random.choice(2,1) == 2:
            r_put = requests.put(GATEWAY_URL + endpoint_put + product_id, auth=auth, headers=headersIAS, data=payload_true)
        else:
            r_put = requests.put(GATEWAY_URL + endpoint_put + product_id, auth=auth, headers=headersIAS, data=payload_false)

def get_suggestions():
    # hitting this endpoint
    endpoint_suggestions = "/catalog/items/suggestions?"
    query = "descriptionPattern=corn+flakes"
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

    status = isAvailable(itemId)

    print(status)

def get_catalog():
    # hitting this endpoint
    endpoint_snapshots = "/catalog/items?"
    # get everything
    query = ""
    # connecting to the API and returning json based on query
    r = requests.get(GATEWAY_URL + endpoint_snapshots + query, auth=auth, headers=headers)
    print(r.content)
    if r.status_code > 400:
        return None
    else: 
        global catalog
        catalog = json.loads(r.content)
    sendData(catalog)

def isAvailable(itemId):
    # hitting this endpoint
    endpoint_available = "/ias/item-availability/"
    # request availability of this itemID
    product_id = itemId.get('itemCode')
    # connecting to the API and returning json based on query
    r = requests.get(GATEWAY_URL + endpoint_available + product_id, auth=auth, headers=headersIAS)
    print(r.content)
    if r.status_code > 400:
        return None
    else:
        isItThere = None
        isItThere = json.loads(r.content)

    if isItThere['isAvailableForSale']:
        return True
    else: 
        return False
    

def sendData(data):
    isSnapshot = 'snapshot' in data
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 10001))
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        while True:
            if not isSnapshot:
                for product in data['pageContent']:
                    resp = json.dumps({"name": product['shortDescription']['values'][0]['value'], "itemId": product['itemId']})
                    print(resp)
                    conn.sendall(resp)
                    time.sleep(2.0)
            else:
                for snap in data['snapshot']:
                    resp = json.dumps({"name": snap['shortDescription']['values'][0]['value'], "itemId": snap['itemId']})
                    print(resp)
                    conn.sendall(resp)
                    time.sleep(2.0)

if __name__ == '__main__':
    prepData()