from urllib.request import urlopen
import json
import pandas as pd


def read_spbom_output(file_name): 
    ########### read in csv#############
    # Load county geojson data
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    # Load and preprocess data
    df = pd.read_csv(file_name, dtype={"FIPS": str})
    df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)

    # Step 1: Create a mapping from FIPS to 'STATE' and 'COUNTY'
    state_code_to_name = {
        "01": "Alabama",
        "02": "Alaska",
        "04": "Arizona",
        "05": "Arkansas",
        "06": "California",
        "08": "Colorado",
        "09": "Connecticut",
        "10": "Delaware",
        "11": "District of Columbia",
        "12": "Florida",
        "13": "Georgia",
        "15": "Hawaii",
        "16": "Idaho",
        "17": "Illinois",
        "18": "Indiana",
        "19": "Iowa",
        "20": "Kansas",
        "21": "Kentucky",
        "22": "Louisiana",
        "23": "Maine",
        "24": "Maryland",
        "25": "Massachusetts",
        "26": "Michigan",
        "27": "Minnesota",
        "28": "Mississippi",
        "29": "Missouri",
        "30": "Montana",
        "31": "Nebraska",
        "32": "Nevada",
        "33": "New Hampshire",
        "34": "New Jersey",
        "35": "New Mexico",
        "36": "New York",
        "37": "North Carolina",
        "38": "North Dakota",
        "39": "Ohio",
        "40": "Oklahoma",
        "41": "Oregon",
        "42": "Pennsylvania",
        "44": "Rhode Island",
        "45": "South Carolina",
        "46": "South Dakota",
        "47": "Tennessee",
        "48": "Texas",
        "49": "Utah",
        "50": "Vermont",
        "51": "Virginia",
        "53": "Washington",
        "54": "West Virginia",
        "55": "Wisconsin",
        "56": "Wyoming"
    }


    fips_to_state_county = {
        feature["properties"]["GEO_ID"][-5:]: {  # Extract the last 5 characters of GEO_ID for FIPS code
            "STATE": feature["properties"]["STATE"],
            "COUNTY": feature["properties"]["NAME"]
        } for feature in counties["features"]
    }

    
    # Step 2: Add 'STATE' and 'COUNTY' to df using the mapping
    df["STATE"] = df["FIPS"].map(lambda x: fips_to_state_county[x]["STATE"] if x in fips_to_state_county else "")
    df["COUNTY"] = df["FIPS"].map(lambda x: fips_to_state_county[x]["COUNTY"] if x in fips_to_state_county else "")
    df["STATE_NAME"] = df["STATE"].map(lambda x: state_code_to_name.get(x, ""))


    
    return df,counties
    
