# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)
from fri.plot import color_palette_3
import matplotlib.mlab as mlab
import numpy as np

import dill

def interactive_scatter_embed(embedding,relevances,classes,mode="markers",txt=None,only_relevant=False):
    relevance_classes = ["Irrelevant", "Weakly relevant", "Strongly relevant"]

    
    size = relevances[:,1]*100

    # Create a trace
    data = []
    for c in [0,1,2]:
        if only_relevant and c == 0:
            print("skipped")
            continue
        if txt is not None:
            text = txt[classes==c]
        else:
            text = np.arange(len(embedding))
        current_class = embedding[classes==c]
        x = current_class[:,0]
        y = current_class[:,1]
        
        trace = go.Scatter(
            x = x,
            y = y,
            name = relevance_classes[c],
            mode = mode,
            text=text,
            hoverinfo = "text",
            marker = dict(
                size = size[classes==c],
                line = dict(
                    width = 3,
                    color = 'rgb(0, 0, 0)')
            )
        )
        data.append(trace)
    layout = go.Layout(
        )
    fig = go.Figure(data=data)

    return fig
    

def relevancebars(relevances,names=None, selected=None):

        print(selected)
        lower = relevances[:,0]
        upper = relevances[:,1]
        if names is None:
            index = np.arange(len(lower))
        else:
            index = names
            
        trace1 = go.Bar(
                x=index,
                y=lower,
                name='Lower Bounds',
                selected=selected
        )
        trace2 = go.Bar(
            x=index,
            y=upper,
            name='Upper Bounds'
        )

        data = [trace1, trace2]
        layout = go.Layout(
            barmode='group'
        )

        fig = go.Figure(data=data, layout=layout)
        return fig


def feature_app(fri_model, feature_names=None):

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    
    embedding = fri_model.relevance_var_embedding_
    intervals = fri_model.interval_
    rel_classes = fri_model.relevance_classes_

    scatter = interactive_scatter_embed(embedding,intervals,rel_classes,txt=feature_names)
    barplot = relevancebars(intervals,feature_names)

    app.layout = html.Div(children=[
        html.H1(children='Hello Test2'),
        dcc.Graph(
            id='scatter-embedding',
            figure=scatter
        ),
        dcc.Graph(
            id='relevancebars',
            figure=barplot
        ),
    ])
    
    # @app.callback(
    #     Output('relevancebars', 'figure'),
    #     [Input('scatter-embedding', 'hoverData')])
    # def display_hover_data(hoverData):
    #     selected_points = [point["pointIndex"] for point in hoverData["points"]]
    #     newfig = relevancebars(intervals,feature_names)
    #     print(newfig.data[0].selectedpoints)
    #     newfig.data[0].selectedpoints=selected_points
    #     return newfig

    return app


if __name__ == '__main__':
    from pathlib import Path

    my_file = Path("student_model.dill")
    if my_file.is_file():
        fri_model = dill.load(open(my_file,"rb"))
    else:
        from fri.genData import genRegressionData
        X,y = genRegressionData(n_samples=100, n_features=6, n_strel=2, n_redundant=2,
                                n_repeated=0, random_state=123)

        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        from fri import FRIRegression
        fri_model = FRIRegression()
        fri_model.fit(X_scaled,y)
        fri_model.grouping()
        fri_model.umap()
        with open("model.dill","wb") as f:
            dill.dump(fri_model,f)

    app = feature_app(fri_model)
    app.run_server(debug=True)