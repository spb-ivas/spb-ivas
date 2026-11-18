import dash
from dash import dcc, html, no_update
import pandas as pd
import plotly.express as px
from urllib.request import urlopen
import json
import plotly.graph_objects as go
import read_spbom_output
from current_prediction import current_prediction_layout3
#from run_model_online_layout_early import run_model_online_layout 
from dash import dcc, html, Input, Output, State
import base64
import io
import subprocess
import os
import glob
from flask import send_file
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import numpy as np
import dash_bootstrap_components as dbc
import re
import ipdb

df,counties=read_spbom_output.read_spbom_output('/root/spbom/spbom_web/output_20250405.default.csv')
def get_file_prefix(file):
    return '_'.join(file.split('_')[:-1])
csv_files = [
    'cpc',
    'fia',
    'prism_30yr_normal',
    'prism_provisional',
    'prism_stable',
    'spots',
    'wassi'
]
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

risk_category_labels = {
    0: "",
    0.5: "0.5 low",
    1.0: "1.0 marginal",
    1.5: "1.5 slight",
    2.0: "2.0 enhanced",
    2.5: "2.5 moderate",
    3.0: "3.0 high"
}


risk_category_colorscale = [
    [0, '#b5f675'],  # low
    [0.16667, '#ffffcc'],  # low
    [0.33333, '#f7ec7a'],  # marginal
    [0.5, '#f9c66b'],  # slight
    [0.66667, '#f28c38'],  # enhanced
    [0.83333, '#e34517'],  # moderate
    [1.0, '#b31717']   # high
]


faq_data = [
    {
        "question": "What is the Southern Pine Beetle Interactive Visualization and Analysis System (SPB IVAS)?",
        "answer": html.Div([
            html.P([
                "The SPB IVAS is a web-based application for visualizing and analyzing data about SPB infestations. The application provides a user-friendly interface for exploring historical data and current year predictions, running the model under different scenarios (i.e., temperature, rainfall, and stocking density change), and visualizing the results on an interactive map."
            ]),
        ])
    },
    {
        "question": "What are the system requirements for using the application?",
        "answer": html.Div([
            html.P("To use the application, ensure that your system meets the following requirements:"),
            html.Ul([
                html.Li([
                    "Web browser: The application is compatible with modern browsers such as Google Chrome, Mozilla Firefox, Microsoft Edge, and Safari."
                ]),
                html.Li([
                    "Internet connection: A stable internet connection is required to access the application and load the necessary data and resources."
                ])
            ]),
        ])
    },
    {
        "question": "How do I access the application?",
        "answer": html.Div([
            html.P("You can access the SPB IVAS at spb.digiapp.org"),
        ])
    },

    {
    "question": "What are the main functions of the Southern Pine Beetle Interactive Visualization and Analysis System?",
    "answer": html.Div([
        html.P("The Southern Pine Beetle  Interactive Visualization and Analysis System offers the following main functions:"),
        html.Ul([
            html.Li([
                "Default display: When the website loads, the display defaults to the current year's SPB predictions by risk category. This allows you to immediately see the predicted risk for the current year without any additional input."
            ]),
            html.Li([
                "Variable selection: Users can choose to display a wide range of variables related to the current year's predictions as well as historical data after 1990. This functionality enables you to explore different aspects of SPB infestation, such as SPB spots, pine density, air temperature, rainfall, and more, across multiple years."
            ]),
            html.Li([
                "Scenario planning: You can run the model to predict the current year SPB infestation under different climate- and management-related scenarios, (i.e., changes in temperature, rainfall, and stocking density. After running the model, you can either download the output data or display the results directly within the system for immediate analysis."
            ]),
            html.Li([
                "User-defined input: The system allows users to upload their own input data to generate customized SPB predictions. You can provide data files ( CSV files required), containing information on SPB spots, pine density, evapotranspiration, and other relevant variables. After uploading the data, you can run the prediction model and either download the output or visualize the results within the system interface."
            ])
        ]),
        html.P([
            "These main functions collectively provide a comprehensive tool for analyzing and predicting SPB infestations. You'll be able to explore historical data, assess current year predictions, simulate climate- and management-related scenarios, and generate predictions based on your input data."
        ]),
    ])
    },

    {
    "question": "What is the Control Panel in the Dashboard?",
    "answer": html.Div([
        html.P("The Control Panel, located on the left side of the Dashboard, provides various options and settings to customize the data visualization:"),
        html.Ul([
            html.Li([
                "Select a variable to plot: Use this drop-down menu to select the variable you want to visualize on the map. Available variables include risk category, SPB spots, pine density, evapotranspiration, and more. ",
                html.Strong(" 1", style={'color': 'red'})
            ]),
            html.Li([
                "Select year to plot: Use this dropdown menu to view variables of interest in the SPBOM1 database. The available years will depend on the historical data provided by the application.",
                html.Strong(" 2", style={'color': 'red'})
            ]),
            html.Li([
                "Search county: Enter the name of a specific county in this search box to highlight it on the map. As you type, the application will provide suggestions based on the entered text.",
                html.Strong(" 3", style={'color': 'red'})
            ]),
            html.Li([
                "Lock colorbar: Check this box to keep the colorbar range fixed across different selections. When locked, the colorbar will maintain a consistent range, allowing for better comparison of values across years or variables.",
                html.Strong(" 4", style={'color': 'red'})
            ]),
            html.Li([
                "Plot difference from default: Use this option to easily compare the output using your data and the default output.",
                html.Strong(" 5", style={'color': 'red'})
            ]),
            html.Li([
                "Data Sources: Use this option to select the data source you want to visualize in the map view. You can choose between the official data (default) or user-defined data.",
                html.Strong(" 6", style={'color': 'red'})
            ]),
            html.Li([
                "Run Model (20250405): Expand this section to run the SPB model with different temperature, rainfall, and pine density scenarios. Adjust the sliders to set the desired scenarios and click the 'Run Model' button to initiate the model run. Once the model run is complete, a download link will appear, allowing you to download the model output file. Click the Display Output button to visualize the model results on the map.",
                html.Strong(" 7", style={'color': 'red'})
            ]),
            html.Li([
                "Run user-defined revisions: Expand this section to run the SPB model with your own input files. Once the model run is complete, a download link will appear, allowing you to download the model output file. Click the Display Output button to visualize the model results on the map.",
                html.Strong(" 8", style={'color': 'red'})
            ])
        ]),
        
        html.Img(src="/assets/control_panel.png", style={'maxWidth': '100%', 'marginTop': '10px'})
    ])
    },

    {
    "question": "What is the Map View in the Dashboard?",
    "answer": html.Div([
        html.Ul([
            html.Li([
                "The Map View  displays the selected variable for each county using a color scale. Hover over a county on the map to see a tooltip with additional information, such as the county name, state, and the value of the selected variable.",
                html.Strong("1", style={'color': 'red'})
            ]),
            html.Li([
                "The map features a legend or colorbar that visually represents the range of values for the plotted variable, providing a clear understanding of data distribution and aiding in the interpretation of spatial patterns.",
                html.Strong("2", style={'color': 'red'})
            ]),

            html.Li([
                "Additional time series analysis is provided, including the time series of the selected variable from 1990 onward and a probability density function of it. ",
                html.Strong("3", style={'color': 'red'})
            ])
        ]),
        html.Img(src="/assets/main_map.png", style={'maxWidth': '100%', 'marginTop': '10px'})
    ])
    },

    {
    "question": "How do I run the SPB model with different scenarios?",
    "answer": html.Div([
        html.Ul([
            html.Li([
                "In the Control panel under 'Run Model 20250405', adjust the temperature, rainfall, and/or pine stocking scenarios to see how those changes impact outbreak risk.",
                html.Strong("1", style={'color': 'red'}),
    ", ",
    html.Strong("2", style={'color': 'red'}),
    ", and ",
    html.Strong("3", style={'color': 'red'})
            ]),
            html.Li([
                "Input a unique name for your model simulation.",
                html.Strong("4", style={'color': 'red'})
            ]),
            html.Li([
                "Click the Run Official Forecast button. The climate scenario model will run, and you will see a 'Running' animation under the 'Model Status' section.",
                html.Strong("5", style={'color': 'red'})
            ]),
            html.Li([
                "After the model run is complete, click the 'Display Output' button.",
                html.Strong("6", style={'color': 'red'})
            ]),
            html.Li([
                "The Map View will now change to display the results of the user-defined model run, with all the associated functions.",
                html.Strong("7", style={'color': 'red'})
            ]),
            html.Li([
                "Toggle between the official default model run and the user-defined model run 8 to compare results.",
                html.Strong("8", style={'color': 'red'})
            ])
        ]),
        html.Img(src="/assets/run_climate.png", style={'maxWidth': '100%', 'marginTop': '10px'})
    ])
    },
    
    {
        "question": "Can I use my own data in the application?",
        "answer": html.Div([
            html.P("Yes, you can use your own data to run the SPB model and visualize the results using the Run User-defined Revision function in the Control Panel. To do so, follow these steps:"),
            html.Ul([
                html.Li([
                    "Format and save your data as CVS files. See the SPBOM1 users guide for data formatting guidance."
                ]),
                html.Li([
                    "Enter a unique name for your data. This name will be used to identify your data and differentiate it from other user data. Once you have entered the name, the upload functions will appear.",
                    html.Strong("1", style={'color': 'red'})
                ]),
                html.Li([
                    "Use the file upload options to select and upload the corresponding data files from your local machine. Make sure to upload all the required files for the model to run successfully.",
                    html.Strong("2", style={'color': 'red'})
                ]),
                html.Li([
                    "After uploading all the data files, click the 'Run User-defined Forecast' button to initiate the model run using your data.",
                    html.Strong("3", style={'color': 'red'})
                ]),
                html.Li([
                    "The application will process your data and generate the model output based on the uploaded files."
                ]),
                html.Li([
                    "After the model run is complete, click the 'Display User-defined Output' button to visualize your results on the map and the associated charts.",
                    html.Strong("4", style={'color': 'red'})
                ])
            ]),
            html.Img(src="/assets/run_own.png", style={'maxWidth': '100%', 'marginTop': '10px'})
        ])
    },
    {
        "question": "What should I do if I encounter issues while using the application?",
        "answer": html.Div([
            html.P("If you encounter any issues while using the application, try the following troubleshooting steps:"),
            html.Ul([
                html.Li([
                    "Refresh the ",
                     "web",
                    " page to reload the application and its components."
                ]),
                html.Li("Clear your browser cache and cookies, then try accessing the application again."),
                html.Li([
                    "Make sure you have a stable ",
                    "internet",
                    " connection to ensure smooth data loading and processing."
                ]),
                html.Li([
                    "Verify that you are using a compatible ",
                    "web",
                    " browser and that it is updated to the latest version."
                ]),
                # Create a list item with the email addresses in bold
                html.Li([
                    "If the issue persists, contact ",
                    html.Strong("steven.mcnulty@usda.gov"),
                    " and ",
                    html.Strong("bosong@clemson.edu")
                ])
            ]),
        ])
    },
#    {
#        "question": "How can I provide feedback or get support for the application?",
#        "answer": html.Div([
#            html.P([
#                "We value your feedback and are committed to continuously improving the Southern Pine Beetle  Interactive Visualization and Analysis System. If you have any suggestions, feature requests, or want to report a bug, please follow these steps:"
#            ]),
#            html.Ol([
#                html.Li("Navigate to the Feedback and Support section of the application."),
#                html.Li("Fill out the provided feedback form, including your name, email address, and a detailed description of your feedback or issue."),
#                html.Li([
#                    "Click the ",
#                    html.Strong("'Submit'"),
#                    " button to send your feedback to the application administrator."
#                ])
#            ]),
#            html.P([
#                "Alternatively, you can contact the application administrator directly via email or any other provided contact information."
#            ]),
#            html.P([
#                "We appreciate your input and will strive to address your concerns and incorporate your suggestions to enhance the user experience and functionality of the application."
#            ]),
#            html.Img(src="/assets/clemson.png", style={'maxWidth': '100%', 'marginTop': '10px'})
#        ])
#    }
]



#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# --- Flask download routes for run-output CSVs (added 2026-05-11) ---
from flask import send_from_directory as _send_from_directory, abort as _abort
_OUTPUT_DIR = '/root/spbom/spbom_web'
_TOKEN_RE = re.compile(r'^[A-Za-z0-9_.-]+$')

@app.server.route('/download/output/<revision>')
def _download_output(revision):
    if not _TOKEN_RE.match(revision):
        _abort(400)
    candidates = [
        'output_{}.csv'.format(revision),
        'output_{}.default.csv'.format(revision),
        'output_{}_None.csv'.format(revision),
    ]
    for fname in candidates:
        if os.path.isfile(os.path.join(_OUTPUT_DIR, fname)):
            return _send_from_directory(_OUTPUT_DIR, fname, as_attachment=True)
    _abort(404)

@app.server.route('/download/output/<revision>/<suffix>')
def _download_output_suffix(revision, suffix):
    if not _TOKEN_RE.match(revision) or not _TOKEN_RE.match(suffix):
        _abort(400)
    fname = 'output_{}_{}.csv'.format(revision, suffix)
    if not os.path.isfile(os.path.join(_OUTPUT_DIR, fname)):
        _abort(404)
    return _send_from_directory(_OUTPUT_DIR, fname, as_attachment=True)
# --- end Flask download routes ---
 #,external_stylesheets=external_stylesheets)

from dash import html

def home_layout():
    return html.Div([
        html.Div(id='slide-image-container', style={'textAlign': 'center'}, children=[
            html.Img(id='slide-image', src=slide_contents[0]["img_src"], style={
                'maxWidth': '100%',
                'maxHeight': '400px',
                'objectFit': 'contain'
            }),
        ]),

        html.Div(style={
            'position': 'absolute', 
            'top': '50%', 
            'left': '50%', 
            'transform': 'translate(-50%, -50%)', 
            'width': '80%', 
            'max-width': '800px',
            'display': 'flex',
            'justify-content': 'space-between',
            'align-items': 'center',
            'background-color': 'rgba(255, 255, 255, 0.8)',
            'padding': '10px',
            'border-radius': '10px'
        }, children=[
            html.Button('<', id='prev-slide', n_clicks=0, 
                        style={'font-size': '24px', 'font-weight': 'bold', 'height': '50px', 'width': '50px'}),
            html.Div(id='text-slide-content', children=slide_contents[0]["text"], 
                     style={'text-align': 'center', 'font-size': '24px', 'font-weight': 'bold'}),
            html.Button('>', id='next-slide', n_clicks=0, 
                        style={'font-size': '24px', 'font-weight': 'bold', 'height': '50px', 'width': '50px'}),
        ]),
        dcc.Store(id='current-slide-index', data=0)
    ],style={
        'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'height': '100vh',
        'padding': '20px',
        'background-image': 'linear-gradient(rgba(224, 247, 250, 0.5), rgba(224, 247, 250, 0.5)), url("/assets/PineBeetles.png")',
        'background-size': 'cover',
        'background-position': 'center',
        'background-attachment': 'fixed'})

slide_contents = [
        {"text":f"add some content.....",
     "img_src": "/assets/large_image.jpg"},
    {"text":f"What is the Southern Pine Beetle disease? The Southern Pine Beetle (SPB) disease refers to the infestation and subsequent damage caused by the bark beetle species Dendroctonus frontalis. This beetle is a major pest of pine forests in the southeastern United States. The SPB is a small, reddish-brown to black insect that burrows into the inner bark of pine trees, disrupting the tree's ability to transport water and nutrients. As the beetles tunnel and reproduce, their larvae feed on the and pupate in the living inner bark. As the beetles tunnel they carry blue stain fungus (Ophioastoma  spp. and Ceratocystis spp.)disrupting the tree\'s ability to transport water and nutrients and, , eventually leading to the death of the infested tree.",
     "img_src": "/assets/large_image.jpg"},
    {"text":"What are the symptoms?           The most visible symptoms of a Southern Pine Beetle infestation include yellowing or browning of the tree's needles (foliage), which typically starts at the top of the crown and spreads downward. Other signs include the presence of pitch tubes or resin on the bark, which are formed as the tree attempts to pitch out the invading beetles. Upon closer inspection, you may also notice S-shaped galleries under the bark, created by the adult beetles, and sawdust-like material at the base of the tree.",
     "img_src": "/assets/background.png"}
    ]
from dash.dependencies import Input, Output, State



def instructions_layout():
    return html.Div([
        html.H1('FAQ'),
        html.H1("Welcome to the Southern Pine Beetle Modeling and Predictions Interactive Visualization and Analysis System (SPB IVAS)."),
        html.H1("Here are some frequently asked questions to help you navigate and use the application effectively."),
        
        html.Ul([
            html.Li(
                html.Details([
                    html.Summary(faq["question"], style={'fontSize': '30px', 'fontWeight': 'bold'}),
                    html.Div([
                        html.P(faq["answer"], style={'fontSize': '30px'}),
                        html.Img(src=faq.get("image", ""), style={'maxWidth': '100%', 'marginTop': '10px'}) if faq.get("image") else None
                    ])
                ])
            )
            for faq in faq_data
        ], style={'listStyleType': 'none'})
    ], style={
#        'display': 'flex',
#        'flexDirection': 'column',
#        'justifyContent': 'center',
#        'height': '100vh',
#        'padding': '20px',
#        'backgroundSize': 'cover',
#        'backgroundPosition': 'center',
#        'backgroundAttachment': 'fixed'
    })

#def instructions_layout():
#    return html.Div([
#        html.H1('Instructions', className="mt-4"),
#        html.P("Welcome to the Southern Pine Beetle  Interactive Visualization and Analysis System. Here are some frequently asked questions to help you navigate and use the application effectively."),
#        
#        html.Ul([
#            html.Li(
#                html.Details([
#                    html.Summary(faq["question"]),
#                    html.P(faq["answer"])
#                ])
#            )
#            for faq in faq_data
#        ], style={'listStyleType': 'none'})
#    ], style={
#        'display': 'flex',
#        'flexDirection': 'column',
#        'justifyContent': 'center',
#        'height': '100vh',
#        'padding': '20px',
#        'backgroundSize': 'cover',
#        'backgroundPosition': 'center',
#        'backgroundAttachment': 'fixed'
#    })
#
def glossary_layout():
    glossary_terms = [
    {'term': 'Southern pine beetle (SPB)',
     'definition': "The southern pine beetle (Dendroctonus frontalis) is a bark beetle species and a major pest of pine forests in the Southeastern United States. The SPB is a small, reddish-brown to black insect that burrows into the inner bark of pine trees to reproduce. The larvae feed on and pupate in the living inner bark. As the beetles tunnel, they carry blue stain fungus (Ophioastoma spp. and Ceratocystis spp.) that disrupts the tree's ability to transport water and nutrients and eventually leads to the death of the infested tree."},
    {'term': 'Symptoms of SPB Infestation',
     'definition': 'The most visible symptoms of an SPB infestation include fading green, yellowing, or reddish brown needles (foliage) that typically starts at the top of the crown and spreads downward. Other signs include the presence of resin pitch tubes (small white or brownish balls of resin) between bark plates. If the tree has adequate soil moisture, pitch tubes, are formed as the tree attempts to pitch out the invading beetles. Upon closer inspection, you may also notice S-shaped galleries under the bark, created by the adult beetles, and small holes in the bark plates where the young beetles have emerged. You may also see sawdust-like material at the base of the tree.'},
    {'term': 'Evapotranspiration',
     'definition': 'Evapotranspiration is a critical ecological process involving the transfer of water from the Earth to the atmosphere through evaporation from soil and other surfaces and by transpiration from plants. In the context of forestry, it is crucial for modeling water usage and stress on plants, particularly during the growing season, which is defined as April to October in temperate regions. This measure helps in assessing forest health and susceptibility to pests like the SPB.'},
    {'term': 'Tmax',
     'definition': 'Tmax represents the maximum temperature observed during a given period. In ecological and agricultural models, Tmax is crucial for understanding and predicting the impacts of climate conditions on vegetation growth and pest development cycles. For forestry applications, it is typically averaged during the growing season (April to October) to assess thermal stress on forests and potential impacts on pest populations like the SPB.'},
    {'term': 'Ownership',
     'definition': 'Ownership refers to the legal or institutional entity that manages land and its natural resources. In the context of forest management, entities like the USDA Forest Service (USDA/FS) are significant as they manage national forests and grasslands across the United States, oversee conservation efforts, and conduct research in sustainable forestry practices, including studies on pests like the SPB.'},
    {'term': 'Risk Category Model',
     'definition': 'The Risk Category Model is a statistical tool used to assess the likelihood of SPB infestation in pine forests. This model processes SPB spot data to categorize risk levels from low to high based on the mean and standard deviation of observed SPB spots. Categories are determined by incremental thresholds set at half, one, one and a half, two, two and a half, and three standard deviations above the mean, representing increasing levels of infestation risk.'},
    {'term': 'Probability Density Function (PDF)',
     'definition': 'In the context of Southern Pine Beetle (SPB) modeling, a Probability Density Function (PDF) is used to represent the likelihood of different risk levels occurring within a given area. The x-axis of the PDF represents the risk levels, while the y-axis represents the density or frequency of these risks. This function helps in visualizing and understanding the distribution of SPB infestation risks across different regions. By analyzing the PDF, forest managers can identify areas with higher probabilities of infestation and prioritize them for monitoring and intervention. The PDF provides a statistical foundation for making informed decisions regarding pest management and resource allocation.'}
    ]
    return html.Div([
        html.H2('Key terms', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Strong(term['term']),
                html.P(term['definition'], style={'textAlign': 'justify'})
            ], style={'marginBottom': '20px'})
            for term in glossary_terms
        ], style={'width': '80%', 'margin': 'auto', 'color':'black', 'font-size':'25px'})
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'flex-start',
        'alignItems': 'center',
        'padding': '20px',
        'overflowY': 'auto',  # Allow scrolling if content overflows vertically
        'maxHeight': '100vh'  # Ensure the content does not exceed the viewport height
    })
    

    

@app.callback(
    [Output(f'label-{file}', 'children') for file in csv_files],
    [Input('user-revision', 'value')]
)
def update_labels(user_revision):
    if user_revision:
        return [f'Upload {file}_{user_revision}.csv:' for file in csv_files]
    else:
        return [f'Upload {file}_{{user_revision}}.csv:' for file in csv_files]

    
@app.callback(
    Output('county-search-dropdown', 'options'),
    Input('county-search-dropdown', 'search_value')
)
def update_county_options(search_value):
    if not search_value:
        return []
    # Filter the counties based on the search value
    filtered_counties = [
        {
            'label': f"{county['properties']['NAME']}, {state_code_to_name.get(county['properties']['STATE'], 'Unknown')}",
            'value': county['properties']['GEO_ID'][-5:]
        }
        for county in counties['features']
        if search_value.lower() in county['properties']['NAME'].lower()
    ]
    return filtered_counties



@app.callback(
    [Output('server-file-container', 'style'),
     Output('server-file-dropdown', 'options')],
    [Input('data-source-radio', 'value')]
)
def update_server_file_dropdown(selected_value):
    if selected_value == 'server':
        server_files = glob.glob('/root/spbom/spbom_web/output_*.csv')
        options = [{'label': file.split('/')[-1], 'value': file} for file in server_files]
        return {'display': 'block'}, options
    else:
        return {'display': 'none'}, []


@app.callback(
    [Output('run-script-button', 'style'),
     Output('run-script-button', 'disabled')],  # Add this line to output the disabled property
    [Input('upload-wassi', 'contents')]
)
def update_run_button_style(wassi_content):
    if wassi_content is not None:
        return {'margin-bottom': '20px', 'background-color': '#1f77b4', 'font-size': '15px',
                'color': 'white', 'border': 'none', 'padding': '10px', 'border-radius': '5px',
                'cursor': 'pointer','margin-right':'10px'}, False
    else:
        return {'margin-bottom': '20px', 'background-color': '#1f77b4', 'font-size': '15px',
                'color': 'white', 'border': 'none', 'padding': '10px', 'border-radius': '5px',
                'cursor': 'not-allowed', 'opacity': '0.5','margin-right':'10px'}, True
    
@app.callback(
    [Output('slide-image', 'src'), Output('text-slide-content', 'children')],
    [Input('prev-slide', 'n_clicks'), Input('next-slide', 'n_clicks')],
    [State('slide-image', 'src'), State('text-slide-content', 'children')]
)
def update_slide(prev_clicks, next_clicks, current_img_src, current_text):
    # Get the current index based on the current image source or text
    current_index = next((index for (index, d) in enumerate(slide_contents) if d["text"] == current_text), 0)
    
    # Check which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks yet'

    # Determine the new index
    if button_id == 'next-slide':
        current_index = (current_index + 1) % len(slide_contents)
    elif button_id == 'prev-slide':
        current_index = (current_index - 1) % len(slide_contents)

    # Get the new slide contents
    new_slide = slide_contents[current_index]

    return new_slide["img_src"], new_slide["text"]


global processed_df_uploaded
processed_df_uploaded = None

def process_uploaded_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

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
        feature["properties"]["GEO_ID"][-5:]: {
            "STATE": feature["properties"]["STATE"],
            "COUNTY": feature["properties"]["NAME"]
        } for feature in counties["features"]
    }

    df_uploaded['FIPS'] = df_uploaded['FIPS'].astype(str).str.zfill(5)
    
    if 'STATE' not in df_uploaded.columns:
        df_uploaded.loc[:, 'STATE'] = df_uploaded['FIPS'].astype(str).str[:2]
    if 'COUNTY' not in df_uploaded.columns:
        df_uploaded.loc[:, 'COUNTY'] = df_uploaded['FIPS'].astype(str).str[2:].astype(int)
    if 'STATE_NAME' not in df_uploaded.columns:
        df_uploaded.loc[:, 'STATE_NAME'] = df_uploaded['STATE'].map(lambda x: state_code_to_name.get(x, ""))

    return df_uploaded

def process_server_data(file_path):
    df_server = pd.read_csv(file_path)

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
        feature["properties"]["GEO_ID"][-5:]: {
            "STATE": feature["properties"]["STATE"],
            "COUNTY": feature["properties"]["NAME"]
        } for feature in counties["features"]
    }

    df_server['FIPS'] = df_server['FIPS'].astype(str).str.zfill(5)
    
    if 'STATE' not in df_server.columns:
        df_server.loc[:, 'STATE'] = df_server['FIPS'].astype(str).str[:2]
    if 'COUNTY' not in df_server.columns:
        df_server.loc[:, 'COUNTY'] = df_server['FIPS'].astype(str).str[2:].astype(int)
    if 'STATE_NAME' not in df_server.columns:
        df_server.loc[:, 'STATE_NAME'] = df_server['STATE'].map(lambda x: state_code_to_name.get(x, ""))

    return df_server


app.layout = html.Div([
#####
    html.Div([
        html.Div([
            html.Img(src='/assets/usdajpg.jpg', style={'display': 'inline-block', 'margin-left': '10px', 'vertical-align': 'middle','width':'auto','height':'100px'}),

            html.H1(['Southern Pine Beetle ',html.Br(),'Interactive Visualization and Analysis System'], style={'display': 'inline-block', 'margin': '0', 'vertical-align': 'middle', 'font-size': '30px', 'font-weight': 'bold', 'color': 'black', 'padding-top': '1px', 'padding-bottom': '1px'}),
            html.Img(src='/assets/clemson.png', style={'display': 'inline-block', 'margin-right': '10px', 'vertical-align': 'middle', 'width':'auto', 'height':'80px'}),

        ], style={'background-color': 'rgba(144, 238, 144, 0.5)', 'padding': '1px', 'text-align': 'center'}),

    html.Div([
    dcc.Link('Dashboard', href='/predictions', style={'margin-right': '10px','fontSize': '22px', 'fontWeight': 'bold'}),
    dcc.Link('Key Terms', href='/glossary', style={'marginRight': '10px', 'fontSize': '22px', 'fontWeight': 'bold'}),
    dcc.Link('FAQ', href='/Instruction', style={'marginRight': '10px', 'fontSize': '22px', 'fontWeight': 'bold'}),
    dcc.Link('Contact:steven.mcnulty@usda.gov', href='mailto:steven.mcnulty@usda.gov', style={'margin-right': '10px','fontSize': '22px', 'fontWeight': 'bold'}),
    html.Span("*Note: Potential issue with input data of AL 2024 SPOTS.", style={'color': 'red', 'margin-left': '15px', 'fontSize': '16px'}),
], style={'background-color': 'lightgray', 'padding': '1px'}),

    html.Div(id='page-content'),

    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),

    
#    html.Div([
#        dcc.Link('Current Outbreaks', href='/current-outbreaks', style={'margin-right': '10px'}),
#        dcc.Link('News & Updates', href='/news-updates', style={'margin-right': '10px'}),
#        dcc.Link('Photo Gallery', href='/photo-gallery', style={'margin-right': '10px'}),
#        dcc.Link('FAQ', href='/faq', style={'margin-right': '10px'}),
#        dcc.Link('Projects', href='/projects', style={'margin-right': '10px'}),
#        dcc.Link('Research Publications', href='/research-publications')
#    ], style={'background-color': 'lightgray', 'padding': '10px'}),
    ])
#####
])

@app.callback(
    Output('upload-cpc-feedback', 'children'),
    [Input('upload-cpc', 'contents')]
)
def display_cpc_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-fia-feedback', 'children'),
    [Input('upload-fia', 'contents')]
)
def display_fia_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-prism_30yr_normal-feedback', 'children'),
    [Input('upload-prism_30yr_normal', 'contents')]
)
def display_prism_30yr_normal_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-prism_provisional-feedback', 'children'),
    [Input('upload-prism_provisional', 'contents')]
)
def display_prism_provisional_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-prism_stable-feedback', 'children'),
    [Input('upload-prism_stable', 'contents')]
)
def display_prism_stable_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-spots-feedback', 'children'),
    [Input('upload-spots', 'contents')]
)
def display_spots_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    Output('upload-wassi-feedback', 'children'),
    [Input('upload-wassi', 'contents')]
)
def display_wassi_upload_feedback(contents):
    if contents is not None:
        return html.Div('Uploaded', style={'color': 'red'})
    return ''

@app.callback(
    [Output('official-script-output', 'children'),
     Output('official-download-link', 'children'),
     Output('official-download-link', 'style'),
     Output('user-script-output', 'children'),
     Output('user-download-link', 'children'),
     Output('user-download-link', 'style'),
     Output('user-loading', 'loading'),
     Output('user-loading-output', 'children'),
     Output('latest-output-file', 'children'),
     Output('display-output-button', 'style'),
     Output('display-output-button', 'disabled'),
     Output('display-user-output-button', 'style'),
     Output('display-user-output-button', 'disabled')],
    [Input('run-official-script-button', 'n_clicks'),
     Input('run-script-button', 'n_clicks')],
    [State('user-revision', 'value'),
     State('official-revision-input', 'value'),
     State('upload-cpc', 'contents'), State('upload-cpc', 'filename'),
     State('upload-fia', 'contents'), State('upload-fia', 'filename'),
     State('upload-prism_30yr_normal', 'contents'), State('upload-prism_30yr_normal', 'filename'),
     State('upload-prism_provisional', 'contents'), State('upload-prism_provisional', 'filename'),
     State('upload-prism_stable', 'contents'), State('upload-prism_stable', 'filename'),
     State('upload-spots', 'contents'), State('upload-spots', 'filename'),
     State('upload-wassi', 'contents'), State('upload-wassi', 'filename'),
     State('official-tmax-adjust-slider', 'value'),
     State('official-et-adjust-slider', 'value'),
     State('user-tmax-adjust-slider', 'value'),
     State('user-et-adjust-slider', 'value'),
     State('official-pine-density-adjust-slider', 'value'),
     State('user-pine-density-adjust-slider', 'value')]
)
def run_script(official_n_clicks, user_n_clicks, user_revision, official_tag,
               cpc_content, cpc_filename,
               fia_content, fia_filename,
               prism_30yr_normal_content, prism_30yr_normal_filename,
               prism_provisional_content, prism_provisional_filename,
               prism_stable_content, prism_stable_filename,
               spots_content, spots_filename,
               wassi_content, wassi_filename,
               official_tmax_adjust, official_et_adjust,
               user_tmax_adjust, user_et_adjust,
               official_pine_density_adjust, user_pine_density_adjust):

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'run-official-script-button':
        revision = '20250405'
        year = '2025'
        try:
            os.chdir('/root/spbom/spbom_web')
            script_path = '/root/spbom/spbom_web/forecast_no_training.py'
            # Determine the appropriate wassi file based on the selected ET change scenario
            if official_et_adjust == 0:
                wassi_file = f"wassi_{revision}_p0.csv"
            else:
                wassi_file = f"wassi_{revision}_{'n' if official_et_adjust < 0 else 'p'}{abs(official_et_adjust)}.csv"

                # Create a symlink from the selected wassi file to wassi_{revision}.csv
            if os.path.exists(f"data/wassi_{revision}.csv"):
                os.remove(f"data/wassi_{revision}.csv")
            os.symlink(wassi_file, f"data/wassi_{revision}.csv")
            subprocess.run(['python', script_path, revision, year,str(official_tmax_adjust), str(official_et_adjust), str(official_pine_density_adjust)], check=True)
                        # Remove the symlink after the script execution
            os.unlink(f"data/wassi_{revision}.csv")


            output_file = f'/root/spbom/spbom_web/output_{revision}_{official_tag}.csv'
            if official_tag != None:
                os.rename(f'/root/spbom/spbom_web/output_{revision}.csv', output_file)
            else:
                os.rename(f'/root/spbom/spbom_web/output_{revision}.csv', output_file)                
            return [
                f'SPBOM model completed successfully.',
                html.A(f'Download output_{revision}_{official_tag}.csv', href=f'/download/output/{revision}/{official_tag}'),
                {'display': 'block'},
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                '', #dash.no_update,
                output_file,
                {'background-color': '#1f77b4', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'pointer'},
                False,
                dash.no_update,
                dash.no_update
            ]    
        except subprocess.CalledProcessError as e:
            return [
                f'Error occurred while running the script: {str(e)}',
                None,
                {'display': 'none'},
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                '',
                dash.no_update,
                {'background-color': 'grey', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'not-allowed'},
                True,
                dash.no_update,
                dash.no_update
            ]
    elif button_id == 'run-script-button':
        if user_revision and all([cpc_content, fia_content, prism_30yr_normal_content,
                                  prism_provisional_content, prism_stable_content,
                                  spots_content, wassi_content]):
            revision = user_revision
            year = user_revision[:4]
            files = [
                ('cpc', cpc_content, cpc_filename),
                ('fia', fia_content, fia_filename),
                ('prism_30yr_normal', prism_30yr_normal_content, prism_30yr_normal_filename),
                ('prism_provisional', prism_provisional_content, prism_provisional_filename),
                ('prism_stable', prism_stable_content, prism_stable_filename),
                ('spots', spots_content, spots_filename),
                ('wassi', wassi_content, wassi_filename)
            ]
            # Validate file names
            for prefix, content, filename in files:
                expected_filename = f"{prefix}_{revision}.csv"
                if filename != expected_filename:
                    return [dash.no_update, dash.no_update, dash.no_update,
                            f"Invalid file name for {prefix}. Please re-upload {expected_filename}",
                            None, {'display': 'none'}, False, '']

            for prefix, content, filename in files:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                filename = f'{prefix}_{revision}.csv'
                file_path = f'/root/spbom/spbom_web/data/{filename}'
                with open(file_path, 'wb') as f:
                    f.write(decoded)

            try:
                os.chdir('/root/spbom/spbom_web')
                script_path = '/root/spbom/spbom_web/forecast_no_training.py'
                subprocess.run(['python', script_path, revision, year,str(user_tmax_adjust), str(user_et_adjust), str(user_pine_density_adjust)], check=True)
                output_file = f'/root/spbom/spbom_web/output_{revision}.csv'
                return [
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    f'User-SPBOM model completed successfully.',
                    html.A(f'Download output_{revision}.csv', href=f'/download/output/{revision}'),
                    {'display': 'block'},
                    dash.no_update,
                    '', #going',
                    output_file,
                    {'background-color': '#ff7f0e', 'color': 'white', 'margin-left':'10px', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'pointer'},
                    False,
                    {'background-color': '#ff7f0e', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'pointer'},
                    False
                ]
            except subprocess.CalledProcessError as e:
                return [dash.no_update, dash.no_update, dash.no_update,
                        f'Error occurred while running the script: {str(e)}', None, {'display': 'none'}, False, '', dash.no_update,dash.no_update, dash.no_update, dash.no_update,False]
        else:
            return [
                dash.no_update,
                dash.no_update,
                dash.no_update,
                f'User-SPBOM model completed successfully.',
                html.A(f'Download output_{revision}.csv', href=f'/download/output/{revision}'),
                {'display': 'block'},
                dash.no_update,
                '',
                output_file,
                dash.no_update,
                dash.no_update,
                {'background-color': '#ff7f0e', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'pointer'},
                False
            ]
    else:
        raise PreventUpdate

    
@app.callback(
    [Output('data-source-radio', 'value'),
     Output('server-file-dropdown', 'value')],
    [Input('display-output-button', 'n_clicks'),
     Input('display-user-output-button', 'n_clicks')],
    [State('latest-output-file', 'children')]
)
def display_output(official_n_clicks, user_n_clicks, latest_output_file):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'display-output-button' and official_n_clicks > 0 and latest_output_file:
        return 'server', latest_output_file
    elif button_id == 'display-user-output-button' and user_n_clicks > 0 and latest_output_file:
        return 'server', latest_output_file
    else:
        raise PreventUpdate
    

@app.callback(
    Output('upload-containers', 'style'),
    Input('user-revision', 'value')
)
def toggle_upload_containers(user_revision):
    if user_revision:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return dash.dcc.Location(id='url-redirect', pathname='/predictions')
    elif pathname == '/predictions':
        return current_prediction_layout3(app, df, counties)
    elif pathname == '/Instruction':
        return instructions_layout()
    elif pathname == '/glossary':  # Add this condition
        return glossary_layout()
    else:
        return home_layout()
#def display_page(pathname):
#    if pathname == '/predictions':
#        return current_prediction_layout3(app, df, counties)
#    else:
#        return home_layout()

#@app.callback(
#    Output('choropleth-map', 'figure'),
#    [Input('variable-dropdown', 'value'),
#     Input('year-dropdown', 'value'),
#     Input('upload-data', 'contents'),
#     Input('data-source-radio', 'value'),
#     Input('server-file-dropdown', 'value')]
#)    
#

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('variable-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('data-source-radio', 'value'),
     Input('server-file-dropdown', 'value'),
     Input('lock-colorbar-checkbox', 'value'),
     Input('county-search-dropdown', 'value'),
     Input('plot-difference-checkbox', 'value')],
    State('choropleth-map', 'figure')
)    

def update_map(selected_variable, selected_year, contents, data_source,server_file,lock_colorbar, selected_county, plot_difference, current_figure):
    global processed_df_uploaded

    variable_units = {
        'Max April Temperature': '°C',
        'Rain': 'mm',
        'SPB_per_thousand_acre': 'Number',
        'SPB_spots': 'Number',
        'Pine_density': '(?)',
        'Evap_Transporation': 'mm',
        'Actual spots': 'Number',
        'Risk_category':'level'
        # Add units for other variables as needed
    }
    
    if data_source == 'upload' and contents is not None:
        processed_df_uploaded = process_uploaded_data(contents, None)
        df_sel = processed_df_uploaded[processed_df_uploaded['YEAR'] == selected_year]
    elif data_source == 'server' and server_file is not None:
        processed_df_server = process_server_data(server_file)
        df_sel = processed_df_server[processed_df_server['YEAR'] == selected_year]
    else:
        df_sel = df[df['YEAR'] == selected_year]


    if 'plot_difference' in plot_difference:
        # Retrieve the default data for the selected year
        default_data = df[df['YEAR'] == selected_year]

        # Calculate the difference between the current data source and the default data
        if data_source == 'upload' and contents is not None:
            df_sel[selected_variable] = df_sel[selected_variable] - default_data[selected_variable]
        elif data_source == 'server' and server_file is not None:
            df_sel[selected_variable] = df_sel[selected_variable] - default_data[selected_variable]


    

    if data_source == 'upload' and contents is not None:
        revision = processed_df_uploaded['REVISION'].iloc[0]  # Assuming 'REVISION' is a column in the uploaded data
    elif data_source == '2server' and server_file is not None:
        revision = server_file #.split('_')[-1].split('.')[0]  # Extract the revision from the server file name
    elif data_source == 'server' and server_file is not None:
        # Regex to capture the group of digits and underscores following 'output_'
        match = re.search(r'output_(.+)\.csv', server_file)
        if match:
            revision = match.group(1)  # This will be '1_2_3' if the filename matches
        else:
            revision = 'default'  # Default revision if the regex doesn't match
    else:
        default_rev='20250405'
        revision = f'default {default_rev} for {selected_year}'  #
        
    counties_with_data = [fips for fips in df_sel['FIPS'].unique()]
    filtered_counties = [county for county in counties['features'] if f"{county['properties']['STATE']}{county['properties']['COUNTY']}" in counties_with_data]
    
    # Add 'STATE_NAME' column to df_sel if it doesn't exist
    if 'STATE_NAME' not in df_sel.columns:
        df_sel['STATE_NAME'] = df_sel['STATE'].map(lambda x: state_code_to_name.get(x, ""))

    # Calculate global statistics
    mean_value = df_sel[selected_variable].mean()
    median_value = df_sel[selected_variable].median()
    std_value = df_sel[selected_variable].std()
    max_value = df_sel[selected_variable].max()
    min_value = df_sel[selected_variable].min()
    q1 = df_sel[selected_variable].quantile(0.25)
    q3 = df_sel[selected_variable].quantile(0.75)
    iqr = q3 - q1
    skewness = df_sel[selected_variable].skew()
    kurtosis = df_sel[selected_variable].kurtosis()
    cv = (df_sel[selected_variable].std() / df_sel[selected_variable].mean()) * 100


    if lock_colorbar and current_figure:
        # Use the colorbar range from the current figure if available
        range_color = current_figure['layout']['coloraxis']['cmin'], current_figure['layout']['coloraxis']['cmax']
    else:
        # Update the colorbar range based on the current year's and variable's data
        #range_color = (df_sel[selected_variable].quantile(0.05), df_sel[selected_variable].quantile(0.95))
        range_color = (df_sel[selected_variable].min(), df_sel[selected_variable].max())
    
    fig = px.choropleth(
        df_sel,
        geojson={'type': 'FeatureCollection', 'features': filtered_counties},
        locations='FIPS',
        color=selected_variable,
#        color_continuous_scale="Viridis",
        color_continuous_scale= risk_category_colorscale if selected_variable == 'Risk_category' else "Viridis",
        scope="usa",
        labels={selected_variable: selected_variable},
        hover_name='FIPS',
        hover_data={
            selected_variable: True,
            'STATE_NAME': True, 
            'COUNTY': True
        },
        range_color=range_color #(df_sel[selected_variable].quantile(0.05), df_sel[selected_variable].quantile(0.95))
    )

    # Customize hover template
    fig.update_traces(hovertemplate='FIPS=%{hovertext}<br>%{customdata[0]:.2f}, %{customdata[1]}<br>%{customdata[2]}') #: %{customdata[3]})



    # Extract actual spots and predicted spots data for all counties in the selected year
    #    ipdb.set_trace()
    if selected_year==2025:
        actual_spots = df_sel['Actual spots']*np.nan
        predicted_spots = df_sel['SPB_spots']*np.nan
    else:    
        actual_spots = df_sel['Actual spots']*np.nan
        predicted_spots = df_sel['SPB_spots']*np.nan
    
    # Create the scatter trace
    scatter_trace = go.Scatter(
        x=actual_spots,
        y=predicted_spots,
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.7
        ),
        name='',
        xaxis='x2',  # Assign the scatter plot to the secondary x-axis
        yaxis='y2',   # Assign the scatter plot to the secondary y-axis
        showlegend=False,  # Dis
        hoverinfo='skip'  # Disable hover tooltips for the scatter trace
    )
    fig.add_trace(scatter_trace)

    """ if selected_year!=2025:
        # Calculate the regression line
        slope, intercept = np.polyfit(actual_spots, predicted_spots, 1)
        regression_line = slope * actual_spots + intercept
        
        # Create the regression line trace
        regression_trace = go.Scatter(
            x=actual_spots,
            y=regression_line,
            mode='lines',
            line=dict(
                color='red',
                width=2
            ),
            name='', 
            xaxis='x2',  # Assign the regression line to the secondary x-axis
            yaxis='y2',   # Assign the regression line to the secondary y-axis
            showlegend=False,
            hoverinfo='skip'  # Disable hover tooltips for the scatter trace
        )
        # Add the scatter trace and regression line trace to the existing choropleth map figure
        fig.add_trace(regression_trace) """
    
    
    
    fig.update_layout(
        hovermode='closest',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        xaxis2=dict(showticklabels=False),
        yaxis2=dict(showticklabels=False),
        title={
            'text': f"revision:{revision} <br> variable:{selected_variable}",
            'x': 0.3,
            'y': 0.95,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'red'}
        },
        geo=dict(
            scope='usa',
            projection={'type': 'albers usa'},
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            fitbounds="locations",
            bgcolor='rgba(0,0,0,0)',
        )
    )
    # Get the units for the selected variable
    units = variable_units.get(selected_variable, '')

    
    if selected_variable == 'Risk_category':
        fig.update_layout(
            # coloraxis_colorbar=dict(
            #     title=selected_variable,
            #     titlefont={'size': 14, 'family': 'Arial, sans-serif', 'color': '#666'},
            #     tickvals=list(risk_category_labels.keys()),
            #     ticktext=[risk_category_labels[val] for val in risk_category_labels.keys()]
            # )
            coloraxis_showscale=False,
            showlegend=True,
            legend=dict(
                title=dict(text='Risk Categories'),
                yanchor="middle",
                y=0.50,
                xanchor="right",
                x=0.95,
                bgcolor='rgba(255, 255, 255, 0.8)',  # semi-transparent white background
                bordercolor='rgba(0, 0, 0, 0.2)',  # light gray border
                borderwidth=1
            )
        )    
        # Add invisible scatter traces for legend with Viridis colors
        legend_items = [
    (0.0, "Minimal", '#b5f675'),     # #b5f675
    (0.5, "Low", '#ffffcc'),         # #ffffcc
    (1.0, "Marginal", '#f7ec7a'),    # #f7ec7a
    (1.5, "Slight", '#f9c66b'),      # #f9c66b
    (2.0, "Enhanced", '#f28c38'),     # #f28c38
    (2.5, "Moderate", '#e34517'),      # #e34517
    (3.0, "High", '#b31717')           # #b31717
]

        for value, label, color in legend_items:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=label,
                marker=dict(
                    size=14,  # Increased size slightly
                    color=color,
                    symbol='square',  # Changed from 'square' to 'rectangle'
                    line=dict(width=1),  # Add a thin border
                    opacity=1
                ),
                showlegend=True
            ))
    else:
        fig.update_layout(
            coloraxis_colorbar=dict(
#                title=selected_variable,
                title=f"({units})",
                titlefont={'size': 14, 'family': 'Arial, sans-serif', 'color': '#666'}
            )
        )
    fig.update_layout({
    'paper_bgcolor': 'rgba(255,255,255,1)',  # Set background color to white
    'plot_bgcolor': 'rgba(255,255,255,1)',  # Set plot background color to white
    'xaxis': {
        'showgrid': False,  # Remove the gridlines
        'zeroline': False,  # Remove the zero line
        'showticklabels': False  # Remove axis labels
    },
    'yaxis': {
        'showgrid': False,  # Remove the gridlines
        'zeroline': False,  # Remove the zero line
        'showticklabels': False  # Remove axis labels
    },
    'margin': {
        'l': 0, 'r': 0, 't': 0, 'b': 0  # Adjust the margin to be zero on all sides
    }
    })
    # Add shapes to create a border
    fig.add_shape(
        # Rectangle reference to the axes
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="Black",  # Border color
            width=2,  # Border width
        )
    )
    # If you're working with a geo plot, like a choropleth map, you may also want to update the geo layout
    fig.update_geos(
        showframe=True,  # Show the frame around the geo plot
        framecolor="Black"  # Set the frame color
    )

    if selected_county:
        fig.add_trace(
            go.Choropleth(
                geojson=counties,
                locations=[selected_county],
                z=[1],
                colorscale=[[0, 'rgba(255, 0, 0, 1)'], [1, 'rgba(255, 0, 0, 1)']],                
                marker_opacity=1.0,
                marker_line_width=2,
                showscale=False,
                showlegend=False,
                hoverinfo='none'
            )
        )
    return fig

@app.callback(
    [Output('hover-data-timeseries', 'figure'),
     Output('hover-data-pdf', 'figure')],
    [Input('choropleth-map', 'hoverData'), 
     Input('variable-dropdown', 'value'),
     Input('data-source-radio', 'value')]
)
def display_hover_data(hover_data, selected_variable, data_source):
    if hover_data is None:
        return go.Figure(), go.Figure()  # Return empty figures if no hover data
    
    fips = hover_data['points'][0]['location']

    if data_source == 'upload' and processed_df_uploaded is not None:
        county_data = processed_df_uploaded[processed_df_uploaded['FIPS'] == fips]
        all_county_mean = processed_df_uploaded.groupby('YEAR')[selected_variable].median().reset_index()
    else:
        county_data = df[df['FIPS'] == fips]
        all_county_mean = df.groupby('YEAR')[selected_variable].median().reset_index()
    
    state_name = county_data['STATE_NAME'].iloc[0]
    county_name = county_data['COUNTY'].iloc[0]
    
    # Calculate statistics
    data = county_data[selected_variable]
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    
    # Create the time series figure
    county_trace = go.Scatter(
        x=county_data['YEAR'],
        y=county_data[selected_variable],
        mode='lines',
        name=f"County"
    )

    all_county_trace = go.Scatter(
        x=all_county_mean['YEAR'],
        y=all_county_mean[selected_variable],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='All-County Median'
    )

    timeseries_fig = go.Figure(data=[county_trace, all_county_trace])
    timeseries_fig.update_layout(
        title=f"{selected_variable} Time-Series<br>for {fips}, {state_name}, {county_name}",
        xaxis_title='Year',
        yaxis_title=selected_variable,
        width=600,
        height=400,
        legend=dict(
            title_text="",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.95,
            bgcolor='rgba(255, 255, 255, 0.8)',  # semi-transparent white background
            bordercolor='rgba(0, 0, 0, 0.2)',  # light gray border
            borderwidth=1,
            orientation="h"
        )
    )
    
    # Create the PDF figure
    pdf_fig = go.Figure(data=[go.Histogram(x=data, histnorm='probability density', name=selected_variable)])
    pdf_fig.update_layout(
        title=f"Probability Density Function of {selected_variable}<br>for {fips}, {state_name}, {county_name}",
        xaxis_title=selected_variable,
        yaxis_title='Density',
        width=600,
        height=400
    )

    return timeseries_fig, pdf_fig  # Return all three figures

@app.callback(
    Output('upload-container', 'style'),
    Input('data-source-radio', 'value')
)
def toggle_upload_container(selected_value):
    if selected_value == 'upload':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8082)
