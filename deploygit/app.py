import dash
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_canvas import DashCanvas
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.express as px
from dash.dependencies import Input, Output
from pandas import DataFrame
from sqlalchemy import create_engine

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

df  = pd.read_csv("intro_bees.csv")
####Start connecting to Postgres
#engine = create_engine('postgresql+psycopg2://postgres:1234@localhost/test')
#df = pd.read_sql_table("productlist3",engine)
#df = df.drop_duplicates()
####Ending connecting to Postgres

x1 = DataFrame(df['Object ID'].unique())

x2 = DataFrame(df['User ID'].unique())

df1 = df
df1['Content'] = df1['Content type'].astype(str) +(' ')+ df1['Object ID'].astype(str)
df1['Content'].head()

x3 = DataFrame(df1['Content'].unique())

dummies = pd.get_dummies(df1['User ID'])
dummies.shape

df201 = pd.concat([df1,dummies],axis='columns')

df1["Total Duration"].fillna(0, inplace = True)
df1["Total Duration"].astype(int)

df["Total Duration"].fillna(0, inplace = True)
df["Total Duration"].astype(int)


DataFrame(df1['Content'].unique())

#df[df['DIVISION']=='1']
df41 = df[df['Content']==df['Content'].unique()[0]]

df_x = df[df['Content']=='Show 107']

df13 = df['Content'].unique()
df13[0]
df4 = df[df['Content']==(df['Content'].unique()[0])]
# df4['Total Duration']
x = df4['Total Duration'].sum()

final_arr = []
for content in df['Content'].unique(): 
    df3 = df[df['Content']==content]
    # df3['Total Duration']
    x = df3['Total Duration'].sum()
    # print('Total duration for each content in sec ',x)
    final_arr.append(x)

final_arr1 = DataFrame(final_arr,columns=['Total Dur'])

df6 = pd.DataFrame(df['User ID'].unique(),columns=['Unique UserID'])
df7 = df6.to_numpy()
df['User ID'].unique()[0]

df6 = pd.DataFrame(df['User ID'].unique(),columns=['Unique UserID']).to_numpy()

pd.DataFrame(df['User ID'].unique(),columns=['Unique UserID']).to_numpy()[0]

df3 = df[df['User ID']==9025]

type(df7)
df8 = df7.astype(np.int)
#df[df['User ID']==9025]

df['User ID'].unique()[0]

#df_x = df[df['User ID']==9025]
df_x = df[df['User ID']==df['User ID'].unique()[0]]

final_arr5 = []
for user in df['User ID'].unique(): 
    df30 = df[df['User ID']==user]
    #df30 = df[df['User ID']==df['User ID'].unique()[0]]
    filt_df = df30
    filt_df['Views'].head()
    x = filt_df['Views'].sum()
    # print('Total Views UniqueID ',x)
    final_arr5.append(x)

df31 = pd.DataFrame(final_arr5,columns=['Views User ID'])

df34 = pd.DataFrame(df['User ID'].unique(),columns=['Unique UserID'])

final_arr6 = []
for user in df['User ID'].unique(): 
    df30 = df[df['User ID']==user]
    #df30 = df[df['User ID']==df['User ID'].unique()[0]]
    filt_df = df30
    filt_df['Clicks'].head()
    x = filt_df['Clicks'].sum()
    # print('Total Views UniqueID ',x)
    final_arr6.append(x)

df32 = pd.DataFrame(final_arr6,columns=['Clicks User ID'])
df32

final_arr7 = []
for user in df['User ID'].unique(): 
    df30 = df[df['User ID']==user]
    #df30 = df[df['User ID']==df['User ID'].unique()[0]]
    # df30['Total Duration']
    x = df30['Total Duration'].sum()
    # print('Total Dur',x)
    final_arr7.append(x)
df35 = pd.DataFrame(final_arr7,columns=['Total Dur'])

df33 = pd.concat([df31,df32,df34,df35],axis=1)
df33 = df33.set_index('Unique UserID')
filt_dataset = df33

#For User Engagement

df['Total_Dur'] = final_arr1
df['Total_Dur']

df['filt_Content'] = DataFrame(df['Content'].unique())

final_arr2 = final_arr1.to_numpy()

s1 = df['Total_Dur']
s2 = df['filt_Content']
df101 = pd.concat([s1, s2],axis=1)
df102 = df101.dropna()
df103 = df102.nlargest(10,'Total_Dur')

df103.rename(columns={'Total_Dur': 'Total Duration for Each User in min', 'filt_Content': 'Most Popular Content'}, inplace=True)

##########################Individual plots

df21 = pd.DataFrame(columns=['Unique ID'])

df21['Unique ID'] = df1['User ID'].unique()

df22 = pd.DataFrame(dummies)

df23 = pd.concat([df21,df22],axis=1)

final_arr1.to_numpy()

final_arr2 = final_arr1.values.tolist()
final_arr3 = np.asarray(final_arr2)

df['Filt Total Dur'] = final_arr1

df3 = df1[df1['Content']=='Show 107']
df3['Total Duration'].head()
#df3['Total Duration'].sum()

df13 = df1['Content'].unique()
df4 = df1[df1['Content']==(df1['Content'].unique()[0])]
df4['Total Duration']
x = df4['Total Duration'].sum()

final_arr11 = []
for content in df1['Content'].unique(): 
    df30 = df1[df1['Content']==content]
    #df30 = df[df['Content']==df['Content'].unique()[0]]
    #df30 = df[df['Content']==df['Content'].unique()[0]]
    filt_df = df30
    filt_df['Total Duration'].head()
    x = filt_df['Total Duration'].sum()
    # print('Total Duration UniqueContent ',x)
    final_arr11.append(x)

df41 = pd.DataFrame(final_arr11,columns=['Total Duration UniqueID'])

df42 = pd.DataFrame(df1['Content'].unique(),columns=['Content'])

df43 = pd.concat([df42,df41],axis=1)
df43 = df43.set_index('Content')
filt_dataset2 = df43

df_x1 = filt_dataset2.nlargest(10,'Total Duration UniqueID')

filt_dataset2['Total Duration UniqueID'].max()

filt_dataset2.nlargest(10, 'Total Duration UniqueID')

# for Show107 (xaxis),=> Unique UserID(9025), Total Duration(Not Total Dur) 
df203 = df201[df201['Content']=='Show 107']
df204 = df203[df203[9025]==1]

df210 = df1[df1['Content']=='Game 47']
#df201['User ID'].unique()

#df210 = df201[df201['Content']==df201['Content'].unique()[48]]
content_idx = 0
# df_store['Game 47'] = {}
df210 = df1[df1['Content']=='Game 47']

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
      
#         if df210[df210['User ID'].unique()[cols]][row]==1:
        
#             #print('index: ',ind)
#             x_1 = df210['Total Duration'].iloc[ind]
#             print('User ID',df210['User ID'].unique()[cols])
#             print('Total Duration: ', x_1)
            
#         ind=ind+1
# df_store.update(df_test)
df_test1 = df_test.T

df210 = df1[df1['Content']=='Video 11197']

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
      
#         if df210[df210['User ID'].unique()[cols]][row]==1:
        
#             #print('index: ',ind)
#             x_1 = df210['Total Duration'].iloc[ind]
#             print('User ID',df210['User ID'].unique()[cols])
#             print('Total Duration: ', x_1)
            
#         ind=ind+1
# df_test.set_index("User ID", inplace = True) 
df_test1 = df_test.T

df210 = df1[df1['Content']=='Video 11197']

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
      
#         if df210[df210['User ID'].unique()[cols]][row]==1:
        
#             #print('index: ',ind)
#             x_1 = df210['Total Duration'].iloc[ind]
#             print('User ID',df210['User ID'].unique()[cols])
#             print('Total Duration: ', x_1)
            
#         ind=ind+1
# df_test.set_index("User ID", inplace = True) 
df_test1 = df_test.T

df1[df1['Content']==df1['Content'].unique()[0]]

df210 = df1[df1['Content']==df1['Content'].unique()[0]]

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
      
#         if df210[df210['User ID'].unique()[cols]][row]==1:
        
#             #print('index: ',ind)
#             x_1 = df210['Total Duration'].iloc[ind]
#             print('User ID',df210['User ID'].unique()[cols])
#             print('Total Duration: ', x_1)
            
#         ind=ind+1
df_test.set_index("User ID", inplace = True) 
df_test1 = df_test.T

df210 = df1[df1['Content']==df1['Content'].unique()[0]]

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
      
#         if df210[df210['User ID'].unique()[cols]][row]==1:
        
#             #print('index: ',ind)
#             x_1 = df210['Total Duration'].iloc[ind]
#             print('User ID',df210['User ID'].unique()[cols])
#             print('Total Duration: ', x_1)
            
#         ind=ind+1
# df_test.set_index("User ID", inplace = True) 
df_test1 = df_test.T
df_test2 = df_test1.to_numpy()

data = np.array([])
df_test2 = df_test1.to_numpy()

# np.append(data,[df_test2],axis=0)

# np.append(data, [a,b], axis=0)
data = np.array([data,df_test2])
# x = pd.DataFrame(data[2])
# x[0]

np.where(df1['Content'].unique() == 'Game 47')

np.where(df1['Content'].unique() == 'Video 11197')

#df210 = df201[df201['Content']==df201['Content'].unique()[48]]
Content_arr=[]
UserID_arr=[]
TotalDuration_arr=[]
Main_data = []
el = 0
for element in range(len(df1['Content'].unique())):
    
    df210 = df1[df1['Content']==df1['Content'].unique()[element]]

    # print('element',df1['Content'].unique()[element])
    Content_arr.append(df1['Content'].unique()[element])
    
#     for cols in range(len(df210['User ID'].unique())):
#         ind = 0
#         for row in df210[df210['User ID'].unique()[cols]]:
#             if row==1:
#                 #print('index: ',ind)
#                 x_1 = df210['Total Duration'].iloc[ind]
#                 print('User ID',df210['User ID'].unique()[cols])
#                 UserID_arr.append(df210['User ID'].unique()[cols])
#                 print('Total Duration: ', x_1)
#                 TotalDuration_arr.append(x_1)
                
#             ind=ind+1
#     el=el+1
#     print('Content Idx ',el)


Content1 = Content_arr

df_Content_arr = pd.DataFrame(Content_arr,columns=['Content'])

df_UserID_arr = pd.DataFrame(UserID_arr,columns=['UserID'])

df_TotalDuration_arr = pd.DataFrame(TotalDuration_arr,columns=['Total_Duration'])

df501 = pd.concat([df_UserID_arr,df_TotalDuration_arr],axis='columns')
df501 = df1.set_index('User ID')



dict2 = {}
dict2.update( {df1['Content'][0]: df_test1 })

dict2.update({ df1['Content'][90]: df_test1 })
dict2[df1['Content'][0]]

dict1 = {}

for num in range(len(df1['Content'].unique())):
    
    df210 = df1[df1['Content']==df1['Content'].unique()[num]]

    df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]

    #         if df210[df210['User ID'].unique()[cols]][row]==1:

    #             #print('index: ',ind)
    #             x_1 = df210['Total Duration'].iloc[ind]
    #             print('User ID',df210['User ID'].unique()[cols])
    #             print('Total Duration: ', x_1)

    #         ind=ind+1
    df_test.set_index("User ID", inplace = True) 
    df_test1 = df_test.T
    dict1.update( {df1['Content'].unique()[num]: df_test1 })



df210 = df1[df1['Content']==df1['Content'].unique()[0]]

df_test = pd.concat([df210['User ID'],df210['Total Duration']],axis=1)   #df210['User ID'].unique()[0]
df_test.set_index("User ID", inplace = True) 
df_test1 = df_test.T

dict1[df1['Content'].unique()[0]]

var_b = [df1['Content'][0]]
var_a = pd.DataFrame(dict1[df1['Content'][0]])

var_a = pd.DataFrame(dict1[df1['Content'][4]])


df1['Content'][0:5]

var_a = pd.DataFrame(dict1[df1['Content'][0]])


var_a[var_a.columns[:]]
# var_a[var_a.columns[0]][0]
# var_a[var_a.columns[1]][0]

final_arr11 = []
for content in df1['Content'].unique(): 
    df30 = df1[df1['Content']==content]
    #df30 = df[df['Content']==df['Content'].unique()[0]]
    #df30 = df[df['Content']==df['Content'].unique()[0]]
    filt_df = df30
    filt_df['Total Duration'].head()
    x = filt_df['Total Duration'].sum()
    # print('Total Duration UniqueContent ',x)
    final_arr11.append(x)

df41 = pd.DataFrame(final_arr11,columns=['Total Duration UniqueID'])

df42 = pd.DataFrame(df1['Content'].unique(),columns=['Content'])

df43 = pd.concat([df42,df41],axis=1)
df43 = df43.set_index('Content')
filt_dataset2 = df43

df700 = filt_dataset2.nlargest(15, 'Total Duration UniqueID')
#####Start Scatter plot
if "Unique UserName" not in filt_dataset.columns:
    username_arr5 = []
    df33_new2=filt_dataset
    for i in (df33_new2.index):
        username_arr5.append(i)
    filt_dataset.insert(2, "Unique UserName", username_arr5, True) 

hover_text = []
###End Scatter plot
# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("m_logo.jpg"),
                            id="plotly-image",
                            style={
                                "height": "150px",
                                "width": "auto",
                                "margin-bottom": "70px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Muslim Kids TV Dashboard",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "User Engagement Overview", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://www.muslimkids.tv/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        # html.P(
                        #     "As of now, Total Users: {numb1}".format(numb1=len(df1['User ID'].unique())),
                        #     className="control_label",
                        # ),
                        html.P(
                            "This is an analytics app of User Engagement for Muslim Kids TV. This App is connected to the database of Muslim Kids TV app and is updated on real time. ",
                            className="control_label",
                        ),
                        html.P(
                            "Bar chart on the right represents 10 Most Popular Content.",
                            className="control_label",
                        ),
                        html.P(
                            "Bar chart below represents proportion of users for each of the Most Popular Content.",
                            className="control_label",
                        ),
                        html.P(
                            "Scatter plot represts Total no. of Clicks Vs Total no. Views for each User, along with Color that shows Total Duration spend for each User.",
                            className="control_label",
                        ),
                        html.P(
                            "Heat Map on the lower-right of the page shows average Logs of each User per week",
                            className="control_label",
                        ),
                        # html.P(
                        #     "Users Currently Logged On: {numb1}".format(numb1="X")",
                        #     className="control_label",
                        # ),
 
                        html.P(
                            "This is an App that is still in Test phase. If you would like to customize an analytics app for your business in order to utilize the data and recommend personalized products for each customers using AI and Machine Learning Model then please contact sanyat.hoque@gmail.com",
                            className="control_label",
                        ),
                        html.P(
                            "About Me, I have specialized in AI and Machine Learning models for over 5 years. I focused my Masters in the field of tree search and neural networks to apply in different range of problems including business models. My passion is always towards creating and applying machine learning models into making business decisions to make an impact in the real world. My goal is to create an app that will reach a billion+ users who would use them everyday.",
                            className="control_label",
                        ),
                        html.Sub(
                            "Copyright(c) 2020, Sanyat Hoque Productions. All Rights Reserved",
                            className="control_label",
                        ),
                        # dcc.RangeSlider(
                        #     id="year_slider",
                        #     min=1960,
                        #     max=2017,
                        #     value=[1990, 2010],
                        #     className="dcc_control",
                        # ),
                        html.P("", className="control_label"),
                        dcc.RadioItems(
                            id="well_status_selector",
                            # options=[
                            #     {"label": "Customize ", "value": "custom"},
                            # ],
                            value="active",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        # dcc.Dropdown(
                        #     id="well_statuses",
                        #     options=well_status_options,
                        #     multi=True,
                        #     value=list(WELL_STATUSES.keys()),
                        #     className="dcc_control",
                        # ),
                        # dcc.Checklist(
                        #     id="lock_selector",
                        #     options=[{"label": "Lock camera", "value": "locked"}],
                        #     className="dcc_control",
                        #     value=[],
                        # ),
                        # html.P("Filter by well type:", className="control_label"),
                        # dcc.RadioItems(
                        #     id="well_type_selector",
                        #     options=[
                        #         {"label": "All ", "value": "all"},
                        #         {"label": "Productive only ", "value": "productive"},
                        #         {"label": "Customize ", "value": "custom"},
                        #     ],
                        #     value="productive",
                        #     labelStyle={"display": "inline-block"},
                        #     className="dcc_control",
                        # ),
                        # dcc.Dropdown(
                        #     id="well_types",
                        #     options=well_type_options,
                        #     multi=True,
                        #     value=list(WELL_TYPES.keys()),
                        #     className="dcc_control",
                        # ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="well_text"), html.P("As of now, Total Users: {numb1}".format(numb1=len(df1['User ID'].unique())))],
                                    id="wells",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="gasText"), html.P("As of now, Total Contents: {numb2}".format(numb2=len(df1['Object ID'].unique())))],
                                    id="gas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="oilText"), html.P("As of now, Total Hours Viewed: {numb3}".format(numb3=df1['Total Duration'].sum()))],
                                    id="oil",
                                    className="mini_container",
                                ),
                                # html.Div(
                                #     [html.H6(id="waterText"), html.P("Total Hours Viewed: {numb3}".format(numb2
                                #     id="water",
                                #     className="mini_container",
                                # ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="count_graph")],     #Main Graph
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph1")],
                    id="countGraphContainer11",
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="main_graph2")],
                    id="countGraphContainer12",
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="main_graph3")],
                    id="countGraphContainer13",
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="main_graph4")],
                    id="countGraphContainer14",
                    className="pretty_container seven columns",
                ),
                # html.Div(
                #     [dcc.Graph(id="main_graph5")],
                #     id="countGraphContainer15",
                #     className="pretty_container seven columns",
                # ),

                # html.Div(
                #     [dcc.Graph(id="individual_graph")],
                #     id="countGraphContainer2",
                #     className="pretty_container five columns",
                # ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="pie_graph")],
                    id="countGraphContainer3",
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="aggregate_graph")],
                    id="countGraphContainer4",
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)
@app.callback(
    [Output("countGraphContainer","value"),
    Output("count_graph", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"
    df = df103
    fig = px.bar(df, 
                 y='Total Duration for Each User in min', 
                 x='Most Popular Content', 
                 text='Total Duration for Each User in min')

    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',title='10  Most  Popular  Contents in Muslim Kids TV!',)
    figure = fig
    # fig.show()
    return container, fig
# @app.callback(
#     [Output("countGraphContainer1","value"),
#     Output("main_graph", "figure")],
#     [
#         Input("well_status_selector", "value"),
#     ]
# )



#Individual plots begin
@app.callback(
    [Output("countGraphContainer11","value"),
    Output("main_graph1", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"

    i=0
    var_a = pd.DataFrame(dict1[df700.index[i]])
    #dict1[df700.index[0]] # 'Game 47'
    var_a
    x1=[df700.index[i]]   #x=[df1['Content'][0:en][i]]

    numb = 0
    fig = go.Figure(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))
    for numb in range((len(var_a.columns))-1):
        fig.add_trace(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))

    fig.update_layout(barmode='stack',title="Clients using '{fname}'".format(fname = df700.index[i]),
    autosize=False,
    width=300,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),)       
    figure=fig 
    return container, figure
@app.callback(
    [Output("countGraphContainer12","value"),
    Output("main_graph2", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"

    i=1
    var_a = pd.DataFrame(dict1[df700.index[i]])
    #dict1[df700.index[0]] # 'Game 47'
    var_a
    x1=[df700.index[i]]   #x=[df1['Content'][0:en][i]]

    numb = 0
    fig = go.Figure(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))
    for numb in range((len(var_a.columns))-1):
        fig.add_trace(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))

    fig.update_layout(barmode='stack',title="Clients using '{fname}'".format(fname = df700.index[i]),
    autosize=False,
    width=300,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),)       
    figure=fig 
    return container, figure

@app.callback(
    [Output("countGraphContainer13","value"),
    Output("main_graph3", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"

    i=2
    var_a = pd.DataFrame(dict1[df700.index[i]])
    #dict1[df700.index[0]] # 'Game 47'
    var_a
    x1=[df700.index[i]]   #x=[df1['Content'][0:en][i]]

    numb = 0
    fig = go.Figure(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))
    for numb in range((len(var_a.columns))-1):
        fig.add_trace(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))

    fig.update_layout(barmode='stack',title="Clients using '{fname}'".format(fname = df700.index[i]),
    autosize=False,
    width=300,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),)       
    figure=fig 
    return container, figure

@app.callback(
    [Output("countGraphContainer14","value"),
    Output("main_graph4", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"

    i=3
    var_a = pd.DataFrame(dict1[df700.index[i]])
    #dict1[df700.index[0]] # 'Game 47'
    var_a
    x1=[df700.index[i]]   #x=[df1['Content'][0:en][i]]

    numb = 0
    fig = go.Figure(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))
    for numb in range((len(var_a.columns))-1):
        fig.add_trace(go.Bar(x=x1, y=var_a[var_a.columns[numb]]))

    fig.update_layout(barmode='stack',title="Clients using '{fname}'".format(fname = df700.index[i]),
    autosize=False,
    width=300,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),)       
    figure=fig 
    return container, figure


#Individual plots end    
@app.callback(
    [Output("countGraphContainer3", "figure"),
    Output("pie_graph", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"
    hover_text = []

    for index,row in filt_dataset.iterrows():
        hover_text.append(('User ID: {users}<br>'+
                        'Total Duration: {country}<br>'+
                        'Clicks: {lifeExp}<br>'+
                        'Views: {gdp}<br>').format(users=row['Unique UserName'],
                                                country=row['Total Dur'],
                                                lifeExp=row['Clicks User ID'],
                                                gdp=row['Views User ID']))

    filt_dataset['text'] = hover_text
    # filt_dataset_new

    data = [go.Scatter(          # start with a normal scatter plot
        x=filt_dataset['Views User ID'],
        y=filt_dataset['Clicks User ID'],
        name='User', 
        text=filt_dataset['text'],
        mode='markers',
        marker=dict(
    #                size=df['Clicks User ID']/30,
                color=filt_dataset['Total Dur'],
                showscale=True
                ) # set the marker size
    )]

    layout = go.Layout(
        title='Total No.of Views/user vs. Total No.of Clicks/user, Color= Total Duration',  #Marker size= Weight,
        xaxis = dict(title = 'Views/user'), # x-axis label
        yaxis = dict(title = 'Clicks/user'),        # y-axis label
        hovermode='closest',yaxis_type="log",xaxis_type="log"
    )
    fig = go.Figure(data=data, layout=layout)
    # pyo.plot(fig, filename='bubble1.html')
    figure=fig
    return container, figure

@app.callback(
    [Output("countGraphContainer4", "figure"),
    Output("aggregate_graph", "figure")],
    [
        Input("well_status_selector", "value"),
    ]
)
def update_graph(slct_year):
    container = "Showing the data for 2019 :"

    df = pd.read_csv('user_log.csv')

    data = [go.Heatmap(
        x=df['DAY'],
        y=df['LST_TIME'],
        z=df['T_HR_AVG'],
        colorscale='Jet'
    )]

    layout = go.Layout(
        title='Average Hourly visits for each User<br>\
        Muslim Kids TV'
    )
    fig = go.Figure(data=data, layout=layout)
    return container, fig



# Main
if __name__ == "__main__":
    app.run_server(debug=True)
