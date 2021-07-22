import json


R = 8.3144622


def load_gkkr_data():
    with open('data/gkkr.json') as file:
        return json.load(file)