#-----importing libraries------------   --------    --------    
from __future__ import print_function
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import dash_bootstrap_components as dbc
from dash import html
import dash
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
from sklearn import metrics


#============SHAP libraries=================================
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import shap


#============ Model libraries===============================
from tcn import TCN


#loading data for analysis
train= pd.read_csv('eda-dash/data/02_interim/train_FD003.csv')
test=pd.read_csv('eda-dash/data/02_interim/test_FD003.csv')
print(train.head())
columns_to_be_dropped=['Engine_no','Cycle','Altitude','Mach','TRA','T2','P2',
                       'P15','P30','epr','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32','RUL']
indicators= train.columns.difference(columns_to_be_dropped)
indicators= list(indicators)
print(dash.__version__)
engines_count = train.Engine_no.nunique()


#prepare features for explainer
cols_to_drop_test= ['Engine_no','Altitude','Mach','TRA','T2','P2','P15','epr','farB','Nf_dmd', 'PCNfR_dmd','RUL']
test_explain= test.drop(columns=cols_to_drop_test)

#Loading the TCN-LSTM Model
reconstructed_model = keras.models.load_model("tcn_lstm_model/tcn_lstm_fd003",
                                              custom_objects={"TCN":TCN })
processed_test_data= np.load('processed_test_data.npy')
processed_train_data= np.load('processed_train_data.npy')
explainer= shap.DeepExplainer(reconstructed_model,processed_train_data[:100]) #model_lstm_tcn

#explain first 30 example predictions
#explaining each prediction 
shap_values= explainer.shap_values(processed_test_data[:30])
print(shap_values[0][0].shape)
cycle_train= train.copy()
cycle_train= cycle_train[['Engine_no','Cycle']].groupby('Engine_no').max().sort_values(by="Cycle",ascending= True)
#print(cycle_train[1:20])
print(type(cycle_train.index))

# #Computing the RMSE scores
# rul_pred = reconstructed_model.predict(processed_test_data).reshape(-1)
# num_test_windows_list= np.load("num_test_windows_list.npy")
# preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
# mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows)) 
#                              for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
# RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))

#=======Defining SHAP force_plot function===================
def _force_plot_html(base_val, i):

    force_plot_ = shap.force_plot(base_val,shap_values[0][i-1][-1],test_explain.columns,matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "200px", "border": 0})

# #create subplots
# fig, axes = plt.subplots(nrows=len(indicators), ncols=1, figsize=(14, 130))
#     # create and adjust superior title
# fig.suptitle("FD003_train", fontsize=15)
# fig.subplots_adjust(top=.975)
#     # create palette
#     #p = _sns.color_palette("coolwarm", engines_count)
# # iterate over indicators
# for i, indicator in enumerate(indicators):
#         # plot scatter plot of each indicator
#         plt.sca(axes[i])
#         plt.scatter(dataframe.RUL,dataframe.loc[:,indicator],c=dataframe.Engine_no,cmap='CMRmap_r')
#         plt.xlabel("RUL")
#         plt.ylabel(indicator)
#         # invert x-axis
#         axes[i].invert_xaxis()


#creating dash instance
app= dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])
server=app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(html.H1("Turbofan Prediction Dashboard", style={'text-align': 'center'}))
        ]),
        dbc.Row(dbc.Col(html.Br())),
        dbc.Row(dbc.Col(html.H3("Sensor Variations"))),
        dbc.Row(
            dbc.Col(dcc.Dropdown( indicators,value= "T24", id="slct_sensor", multi=False, style={'width': "40%"},
                                 placeholder="Select sensor"))),
        dbc.Row(dbc.Col(html.Br())),
        dbc.Row(dbc.Col(html.Div(id='dd-output-container'))),
        dbc.Row(dbc.Col(dcc.Graph(id='create_sensor_plot',figure={}))),
        # dbc.Row(dbc.Col(html.Br())),
        # dbc.Row(dbc.Col(dcc.Dropdown(options= [{'1-20':20}, {'21-40',40}, {'41-60': 60}, {'61-80':80},{'81-100 ':100}],
        #                              value= 20,multi=False, id= 'slct_engine_range',style={'width':'40%'}, placeholder="Select Engine Range"))),
        # #dbc.Row(dbc.Col(html.H3("The number of cycles per Engine")))
        # dbc.Row(dbc.Col(dcc.Graph(id= 'max_cycles_engine_plot',figure={}))),
        dbc.Row(dbc.Col(html.Br())),
        dbc.Row(dbc.Col(html.H3("SHAP Force Plot"))),
        dbc.Row(dbc.Col(dcc.Dropdown(
                options= [{'label': k, 'value': v} 
                           for k, v in dict(zip(['Sample {}'.format(str(i)) for i in np.arange(1,31,1)],np.arange(1,31,1))).items()
                           ],value=30 , id="slct_engine",multi=False, style={'width': "60%"},placeholder="Select Test sample"
         ))),
         dbc.Row([dbc.Col(html.Div(id='dd-output-container-2'))]),
         dbc.Row(dbc.Col(html.Div(id='force_plot'))),
         dbc.Row(dbc.Col(html.Br()))
    ])


# html.Div( className= 'row',children= [ 
    
#     html.H1("Turbofan Prediction Dashboard", style={'text-align': 'center'}),
#     html.Br(),
#     html.Div(className='row',children= [
#         html.Col(
#             [dcc.Dropdown(
#     indicators,value= "T24", id="slct_sensor", multi=False, style={'width': "40%"},
#     placeholder="Select sensor"
#     ), 
#     html.Div(id='dd-output-container'),
#     dcc.Graph(id='create_sensor_plot', figure={})]),
#     html.Col([
#     dcc.Dropdown(options= [{'label': k, 'value': v} 
#                            for k, v in dict(zip(['Engine {}'.format(str(i)) for i in np.arange(1,31,1)],np.arange(1,31,1))).items()
#                            ],value=30 , id="slct_engine",multi=False, style={'width':"40%"},
#         placeholder="Select engine unit"
#     ),
#     html.Div(id='dd-output-container-2'),
#     html.Div(id='force_plot' )])
#     ])
# ]
# )

#============ Connecting plotly components with dash===============================
@app.callback(
    [
        Output(component_id= 'dd-output-container', component_property= 'children'),
        Output(component_id= 'create_sensor_plot',component_property= 'figure')
    ],
    [Input(component_id='slct_sensor', component_property='value')
    ]
)
def update_graph(option_slcted):
    
    df= train.copy()
    print(option_slcted)
    print(type(option_slcted))
    df= df.groupby(['RUL']).agg(['min', 'mean', 'max'])
    container= "The sensor selected is {} ".format(option_slcted)
    df_grouped_ind = df[str(option_slcted)]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=df_grouped_ind.index,y=df_grouped_ind['max'],mode='lines',name='Max'))
    figure.add_trace(go.Scatter(x=df_grouped_ind.index,y=df_grouped_ind['mean'],mode='lines',name='Mean'))
    figure.add_trace(go.Scatter(x=df_grouped_ind.index,y=df_grouped_ind['min'],mode='lines',name='Max'))
    figure.update_layout(title='Variations of {} against RUL w.r.t Max, Min and Average'.format(option_slcted),
                         xaxis_title='RUL(cycles)',
                   yaxis_title='{}'.format(option_slcted))
    return container,figure

# @app.callback(
#         [
#             Output(component_id='max_cycles_engine_plot',component_property='figure')
#         ],
#         [
#             Input(component_id='slct_engine_range',component_property='value')
#         ]
# )
# def update_cycles_max(slct_engine_range):
#     cycle_train= train.copy()
#     cycle_train= cycle_train[['Engine_no','Cycle']].groupby('Engine_no').max().sort_values(by="Cycle",ascending= True)
#     cycle_train= cycle_train[slct_engine_range-19-1:slct_engine_range]
#     fig_cycles= px.bar(x=cycle_train.index.to_list() ,y= cycle_train.values)
#     # make the figure a bit more presentable
#     fig_cycles.update_layout(title='Engines: {} to {}'.format(slct_engine_range-19,slct_engine_range),
#                              yaxis_title= 'Engine unit number',xaxis_title ='Number of Cycles',cliponaxis=False)
#     return fig_cycles


@app.callback(
    [Output(component_id='dd-output-container-2',component_property='children'),
     Output(component_id='force_plot',component_property='children')
     #Output(component_id='create_shap_plot',component_property='figure'),
    ],
    [
        Input(component_id='slct_engine',component_property='value')
    ]
)
def update_expln_graph(id_selected):
    container= "Test Sample {} is selected.".format(id_selected)
    plot= _force_plot_html(explainer.expected_value[0],id_selected)
    #figure= _force_plot_html(explainer.expected_value[0],id_selected)
    return container ,plot#figure


if __name__ == '__main__' :
    app.run_server(debug=True)
