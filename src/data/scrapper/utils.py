import requests
import time


headers = {"User-agent": "your bot 0.1"}

def repeated_request(url, count = 3):
    time.sleep(2)
    print('-------')
    print('Try to get url:', url)
    response = requests.get(url.strip(), headers)
    if response.status_code != 200 and count > 1:
        print('PAUSE...')
        time.sleep(30 * max(4 - count, 1)) 
        return repeated_request(url, count - 1)
    elif response.status_code != 200 and count == 1:
        print("FAILED: it was last count. code:", response.status_code)
    else:
        print('SUCCESS')
    return response