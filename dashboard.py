import pandas as pd
import joblib
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np

# data and model importation

df = requests.get("http://127.0.0.1:8000/get_data").json()
df = pd.DataFrame(df)

# data processing
df = df.fillna(0)
df_table = df[['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]
data = df.groupby(['TARGET'])['EXT_SOURCE_3'].mean()
data = pd.DataFrame(data).reset_index()
data2 = df.groupby(['TARGET'])['EXT_SOURCE_2'].mean()
data2 = pd.DataFrame(data2).reset_index()
data3 = df.groupby(['TARGET'])['PAYMENT_RATE'].mean()
data3 = pd.DataFrame(data3).reset_index()
IDs = df.SK_ID_CURR.unique()

# dashboard creation

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['AMT_CREDIT'],y=df['AMT_ANNUITY'],mode='markers',opacity=0.5,name='Autres clients'))
fig1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 0,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"}))
fig2 = px.bar(data, x="TARGET", y='EXT_SOURCE_3')
fig3 = px.bar(data2, x="TARGET", y='EXT_SOURCE_2')
fig4 = px.bar(data3, x="TARGET", y='PAYMENT_RATE')
fig5 = go.Figure()

lst = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'ANNUITY_INCOME_PERC', 'REGION_POPULATION_RELATIVE']
for col in lst:
    fig5.add_trace(go.Box(y=df[col].values, name=df[col].name))

fig5.update_layout(title=f"Most important features")

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = html.Div([
    html.Br(),
    html.H1("Home credit dashboard", style={'text-align': 'center'}),
    html.Br(),
    html.H6("Select the client ID"),
    dcc.Dropdown(
        id="IDS_dropdown",
        options=[{"label" : client_id, "value": client_id} for client_id in IDs],
        value="Choose an ID",
        multi=False,
        style={"width": "50%"}),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H6("Credit information"),
            dash_table.DataTable(columns = [{"name": i, "id": i} for i in df_table.columns],
                                 id='table')],width=6)
    ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='score', figure=fig1)], width=4),
        dbc.Col([dcc.Graph(id='my_graph', figure=fig)], width=8)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='my_graph5', figure=fig5), width=12),
        dbc.Col(dcc.Graph(id='my_graph2', figure=fig2), width=4),
        dbc.Col(dcc.Graph(id='my_graph3', figure=fig3), width=4),
        dbc.Col(dcc.Graph(id='my_graph4', figure=fig4), width=4)
    ])
])  

@app.callback(
    Output('table', 'data'),
    Input("IDS_dropdown", "value")
)

def update_table(value):
    dff = df_table[df_table['SK_ID_CURR'] == value]
    return dff.to_dict('records')  


@app.callback(
    Output('score', 'figure'),
    Input('IDS_dropdown', "value")
)

def update_score(client_id):
    r = requests.get("http://127.0.0.1:8000/predict", params={"client_id" : client_id})
    val = r.json()
    if val[0] >= val[1]:
        val = val[0]
        status = 'accepted'
        color = 'green'
    else:
        val = val[1]
        status = 'refused'
        color = 'red'
    
    fig1 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val*100,
        number = {'suffix':'%'},
        gauge = {
            "axis" : {'range' : [None, 100]},
            "steps" : [
                {"range" : [0, 100], "color" : color},
                ],
        },
        domain = {'x': [0, 1], 'y': [0, 1]}))
    
    fig1.update_layout(title=f"The credit is {status} with a score of")

    return fig1

@app.callback(
    [Output('my_graph', 'figure'),
    Output('my_graph5', 'figure'),
    Output('my_graph2', 'figure'),
    Output('my_graph3', 'figure'),
    Output('my_graph4', 'figure')],
    Input("IDS_dropdown", "value")
)

def update_graph(client_id):

    selected_x = df[df['SK_ID_CURR'] == client_id]['AMT_INCOME_TOTAL']
    selected_y = df[df['SK_ID_CURR'] == client_id]['AMT_ANNUITY']

    fig = go.Figure()
    rate = np.around(df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL'],2)*100

    fig.add_trace(go.Scatter(
        x=df['AMT_INCOME_TOTAL'],
        y=df['AMT_ANNUITY'],
        mode='markers',
        marker=dict(
            opacity=0.5
        ),
        text=rate,
        name='Autres clients'))

    fig.add_trace(go.Scatter(
        x=selected_x,
        y=selected_y,
        mode='markers',
        marker=dict(
        color='red',
        opacity=1
    ),
    text=rate, 
    name=f'Client {client_id}'))

    fig.update_traces(hovertemplate='Valeur: %{text}%')

    fig.update_layout(
        xaxis_title='Income amount',
        yaxis_title='Annuity amount',
        title=f"Income vs Annuity - Client ID: {client_id}")
    
    x = df[df['SK_ID_CURR'] == client_id]['TARGET']
    y = df[df['SK_ID_CURR'] == client_id]['EXT_SOURCE_3']
    
    fig2 = px.bar(data, x="TARGET", y='EXT_SOURCE_3', color_discrete_sequence=["#ff9900"])
    fig2.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Client {client_id}'))
    fig2.update_layout(title=f"EXT_SOURCE_3 - Client ID: {client_id}")

    x1 = df[df['SK_ID_CURR'] == client_id]['TARGET']
    y1 = df[df['SK_ID_CURR'] == client_id]['EXT_SOURCE_2']

    fig3 = px.bar(data2, x="TARGET", y='EXT_SOURCE_2')
    fig3.add_trace(go.Scatter(x=x1, y=y1, mode='markers', name=f'Client {client_id}'))
    fig3.update_layout(title=f"EXT_SOURCE_2 - Client ID: {client_id}")

    x2 = df[df['SK_ID_CURR'] == client_id]['TARGET']
    y2 = df[df['SK_ID_CURR'] == client_id]['PAYMENT_RATE']

    fig4 = px.bar(data3, x="TARGET", y='PAYMENT_RATE', color_discrete_sequence=["#b6e880"])
    fig4.add_trace(go.Scatter(x=x2, y=y2, mode='markers', name=f'Client {client_id}'))
    fig4.update_layout(title=f"PAYMENT_RATE- Client ID: {client_id}")
        
    return fig, fig5, fig2, fig3, fig4
      
if __name__ == '__main__':
    app.run_server(debug=True)