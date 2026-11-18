import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from urllib.request import urlopen
import json
from dash.dependencies import Input, Output


def current_prediction_layout3(app, df, counties):
    official_revision = '20250405'
    data_directory = '/root/spbom/spbom_web/data'
    csv_files = [
        f'cpc_{official_revision}.csv',
        f'fia_{official_revision}.csv',
        f'prism_30yr_normal_{official_revision}.csv',
        f'prism_provisional_{official_revision}.csv',
        f'prism_stable_{official_revision}.csv',
        f'spots_{official_revision}.csv',
        f'wassi_{official_revision}.csv'
    ]

    def get_file_prefix(file):
        return '_'.join(file.split('_')[:-1])
    
    layout = html.Div([
#        html.H3("Interactive Visualization and Analysis System", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'color': '#1f77b4'}),
        html.Br(),
        
        html.Div([
            html.Div([
                html.H1("Control Panel", style={'text-align': 'center'}),

                html.Label('Select a variable to plot:', style={'fontSize': '15px', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns if col != 'FIPS' and col != 'YEAR' and col != 'Yt' and col != 'STATE' and col != 'STATE_NAME' and col != 'COUNTY'],
                    value='Risk_category',
                    style={'width': '100%', 'font-family': 'Arial, sans-serif'}
                ),
                html.Label('Select year to plot:', style={'fontSize': '15px', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': year, 'value': year} for year in df['YEAR'].unique()],
                    value=df['YEAR'].max(),
                    style={'width': '100%', 'font-family': 'Arial, sans-serif'}
                ),


                html.Div([
                    html.Label('Search County'),
                    dcc.Dropdown(
                        id='county-search-dropdown',
                        options=[],
                        value=None,
                        placeholder='Enter a county name...',
                        style={'width': '100%'}
                    )
                ], style={'margin-top': '20px'}),
                
                html.Div([
                    html.Label('Lock Colorbar'),
                    dcc.Checklist(
                        id='lock-colorbar-checkbox',
                        options=[{'label': '', 'value': 'lock'}],
                        value=[],
                        style={'margin-left': '10px'}
                    )
                ], style={'margin-top': '20px'}),

                html.Div([
                    html.Label('Plot Difference from Default'),
                    dcc.Checklist(
                        id='plot-difference-checkbox',
                        options=[{'label': '', 'value': 'plot_difference'}],
                        value=[],
                        style={'margin-left': '10px'}
                    )
                ], style={'margin-top': '20px'}),
                
                html.Hr(style={'color': '#000000', 'background-color': '#000000', 'height': '2px'}),
                html.H3("Data Sources", style={'text-align': 'center'}),
                dcc.RadioItems(
                    id='data-source-radio',
                    options=[
                        {'label': 'Official Data', 'value': 'default'},
#                        {'label': 'Own Data (Upload your own data)', 'value': 'upload'},
                        {'label': 'User Data (Select from server)', 'value': 'server'}  # Add this option
                    ],
                    value='default',
                    labelStyle={'display': 'block', 'margin-bottom': '10px', 'fontSize': '15px'},
                    style={'fontSize': '15px'}
                ),
                html.Div([
                    html.Label('Select Server File:'),
                    dcc.Dropdown(
                        id='server-file-dropdown',
                        options=[],
                        value=None,
                        style={'width': '100%', 'font-family': 'Arial, sans-serif'}
                    )
                ], id='server-file-container', style={'display': 'none'}),

                
                html.Div([
                    html.Label('Upload Data:'),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=False
                    )
                ], id='upload-container', style={'display': 'none'}),
                
                html.Hr(style={'color': '#000000', 'background-color': '#000000', 'height': '2px'}),

                html.H3(f'Model Status', style={'text-align': 'center'}),
                html.Div([
                    html.Div(
                        id='loading-container',
                        children=[
                            dcc.Loading(
                                id='user-loading',
                                type='default',
                                children=html.Div(id='user-loading-output'),
                                fullscreen=False,
                                style={'margin-left': '10px', 'display': 'inline-block', 'vertical-align': 'middle'},
                                className='gif-loading'

                            ),
                            html.Div(
                                id='running-text',
                                children="Running!",
                                style={'display': 'none', 'margin-left': '10px', 'font-weight': 'bold', 'color': 'green'}
                            )
                        ]
                    )
                ], style={'margin-bottom': '40px', 'font-size': '15px'}),


#
                html.Hr(style={'color': '#000000', 'background-color': '#000000', 'height': '2px'}),
                html.H3(f'Run Model {official_revision}', style={'text-align': 'center'}),
                html.Div([
                    html.P('Temperature Change Scenario (-2°C to +7°C): Choose 0 for present-day climate'),
                    dcc.Slider(
                        id='official-tmax-adjust-slider',
                        min=-2,
                        max=7,
                        step=1,
                        value=0,
                        marks={i: f'{i}°C' for i in range(-2, 8)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    html.P('Rainfall Change Scenario (-40% to +40%): Choose 0% for present-day climate'),
                    dcc.Slider(
                        id='official-et-adjust-slider',
                        min=-40,
                        max=40,
                        step=5,
                        value=0,
                        marks={i: f'{i}%' for i in range(-40, 41, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.P('Pine Density Adjustment (-20% to +20%): Choose 0% for no change'),
                    dcc.Slider(
                        id='official-pine-density-adjust-slider',
                        min=-20,
                        max=20,
                        step=1,
                        value=0,
                        marks={i: f'{i}%' for i in range(-20, 21, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'margin-bottom': '20px', 'font-size': '15px', 'padding': '0 20px'}),
                html.Div([
                    html.Label('Enter a unique  tag:', style={'fontSize': '15px', 'marginTop': '20px'}),
                    dcc.Input(id='official-revision-input', type='text', placeholder='e.g., Climate_exp', 
                              style={'width': '100%', 'padding': '5px'})
                ], style={'marginBottom': '20px'}),
                
                
                html.Button('Run Official Forecast', id='run-official-script-button', n_clicks=0,
                            style={'margin-bottom': '20px', 'background-color': '#1f77b4', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'pointer','margin-right': '10px'}),
                
                html.Button('Display Output', id='display-output-button', n_clicks=0,
                            style={'margin-bottom': '20px', 'background-color': 'grey', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'not-allowed'},
                            disabled=True),  # Set initial state to disabled

#                html.Button('Display Output', id='display-output-button', n_clicks=0,
#            style={'margin-bottom': '20px', 'background-color': '#1f77b4', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': #'5px', 'cursor': 'pointer'}),
                
                html.Div(id='official-script-output', style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'}),
                html.Div(id='official-download-link', style={'display': 'none'}),
                html.Div(id='latest-output-file', style={'display': 'none'}),  # Add this line
                html.Hr(style={'color': '#000000', 'background-color': '#000000', 'height': '2px'}),

                html.Div([
                    html.H3(f'Run User-defined Revision', style={'text-align': 'center'}),

                    dcc.Input(id='user-revision', type='text', placeholder='Enter your REVISION (e.g., 20250414) Do not use the official revision',
                              style={'margin-bottom': '20px', 'width': '70%', 'padding': '5px', 'border': '1px solid #ccc', 'border-radius': '5px'}),
                    html.H5('Upload Files:', style={'margin-bottom': '10px', 'color': '#7f7f7f'}),
                    html.Div([
                        html.Div([
                            #html.Label(f'Upload {get_file_prefix(file)}_{{user_revision}}.csv:', style={'display': 'block', 'margin-bottom': '5px', 'color': '#2ca02c'}),
                            html.Label(f'Upload {get_file_prefix(file)}_{{user_revision}}.csv:', id=f'label-{get_file_prefix(file)}', style={'display': 'block', 'margin-bottom': '5px', 'color': '#2ca02c'}),
                            dcc.Upload(
                                id=f'upload-{get_file_prefix(file)}',
                                children=html.Div(['Select File']),
                                style={
                                    'width': '20%',
                                    'height': '40px',
                                    'lineHeight': '40px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin-bottom': '10px',
                                    'background-color': '#f0f0f0',
                                    'color': '#2ca02c',
                                    'cursor': 'pointer'
                                }
                            ),
                            html.Div(id=f'upload-{get_file_prefix(file)}-feedback', style={'color': '#9467bd'})
                        ]) for file in csv_files
                    ], id='upload-containers', style={'display': 'none'}),
                ], style={'margin-bottom': '20px'}),
                html.Div([
                    html.P('Temperature Change Scenario (-2 deg to +7 deg): Choose 0 for present-day climate'),
                    dcc.Dropdown(id='user-tmax-adjust-slider',
                                 options=[{'label': str(i), 'value': i} for i in range(-2, 8)],
                                 value=0,
                                 clearable=False,
                                 style={'width': '100%'}),
                    html.P('ET Change Scenario (-40% to +30%): Choose 0% for present-day climate'),
                    dcc.Dropdown(id='user-et-adjust-slider',
                                 options=[{'label': str(i) + '%', 'value': i} for i in range(-40, 31, 5)],
                                 value=0,
                                 clearable=False,
                                 style={'width': '100%'}),
                    html.P('User Pine Density Adjustment (-20% to +20%): Choose 0% for no change'),
                    dcc.Slider(
                        id='user-pine-density-adjust-slider',
                        min=-20,
                        max=20,
                        step=1,
                        value=0,
                        marks={i: f'{i}%' for i in range(-20, 21, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'display': 'none', 'margin-bottom': '20px', 'font-size': '15px'}),
                
                html.Button('Run User Own Forecast', id='run-script-button', n_clicks=0,
                            style={'margin-bottom': '20px', 'background-color': '#1f77b4', 'font-size': '15px',
                                   'color': 'white', 'border': 'none', 'padding': '10px', 'border-radius': '5px',
                                   'cursor': 'not-allowed', 'opacity': '0.5', 'margin-right': '10px'},
                            disabled=True),  # Set the initial disabled property to True
                
                html.Button('Display User Output', id='display-user-output-button', n_clicks=0,
                            style={'margin-bottom': '20px', 'background-color': 'grey', 'color': 'white', 'border': 'none', 'padding': '10px', 'font-size':'15px','border-radius': '5px', 'cursor': 'not-allowed', 'margin-left':'10px'},
                            disabled=True),  # Set initial state to disabled

                html.Div(id='user-script-output', style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'}),
                html.Div(id='user-download-link', style={'display': 'none'})
                
            ], style={'width': '25%', 'padding': '20px', 'background-color': '#f8f9fa', 'box-shadow': '2px 2px 2px lightgrey'}),
                

            
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='choropleth-map',
                        responsive=True,
                        style={'height': '50vh'},
                        config={
                            'displayModeBar': False
                        }
                    )
                ], style={'width': '100%', 'padding': '1px'}),
                html.Div([
                    dcc.Graph(id='hover-data-timeseries'),
                    dcc.Graph(id='hover-data-pdf')
                ], style={'width': '100%', 'display': 'flex', 'flex-wrap': 'wrap'})
            ], style={'width': '75%'}) #,'margin-left': '25%', 'padding-left': '20px','padding-top': '20px'})
        ], style={'display': 'flex'}),
    ])

    return layout
