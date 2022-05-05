import json


def json_convert(x):
    try:
        data = json.loads(x)
    except:
        data = None
    return data
