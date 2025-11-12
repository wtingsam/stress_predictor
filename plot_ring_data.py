# Load a file from json
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_ring_data(data, data_type):
    timestamps = []
    values = []
    _data = data["data"]["metric_data"]
    print(_data) 
    # Loop over all types of data
    for i in range(len(_data)):
        _type = _data[i]["type"]
        obj = _data[i]["object"]
        print(_type)
        if _type == data_type:
            for item in obj["values"]:
                ts = datetime.datetime.fromtimestamp(item["timestamp"])
                print(ts)
                val = item["value"]
                timestamps.append(ts)
                values.append(val)
            break
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values, marker='o', linestyle='-')
    plt.title(f'{data_type.upper()} Data Over Time')
    plt.xlabel('Time')
    plt.ylabel(f'{data_type.upper()} Value')
    plt.grid(True)
    plt.show()

# Use input as a json file path by argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Ring Data from JSON file')
    parser.add_argument('file_path', type=str, help='Path to the JSON file')
    parser.add_argument('data_type', type=str, help='Type of data to plot (hr, temp)')
    args = parser.parse_args()

    data = load_json(args.file_path)
    plot_ring_data(data, args.data_type)
