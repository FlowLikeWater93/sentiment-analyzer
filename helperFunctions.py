import pandas as pd
import numpy as np
import json


def handle_large_jsons(size, offset, filename="yelp_dataset/yelp_academic_dataset_review.json"):
    '''
    Parameters :
    1- size : Number of json objects to parse and add to the Dataframe
    2- offset : line number to start parsing from

    - Open Json as file
    - Go through the file line by line
    - Parse line as JSON
    - Add data to list
    - Append list to df_data as a row

    Returns :
    Pandas dataframe
    '''
    # Data frame data and columns
    df_data = []
    columns = []
    data = None

    # Openning the file
    with open(filename) as file:
        counter = -1
        for line in file:
            counter += 1
            # Only start if line number equals offset
            if counter < offset:
                continue

            # Parse line as json (return dictionary)
            data = json.loads(line.replace('\n', ''))

            # Append dictionary data to list
            row = []
            for col in data:
                row.append(data[col])

            # Append the list to master matrix that will be converted to pandas dataframe
            df_data.append(row)

            # Exit if df_data = size
            if counter == size+offset-1:
                break

    # Return Pandas Dataframe
    columns = data.keys()
    return pd.DataFrame(data=df_data, columns=columns)
