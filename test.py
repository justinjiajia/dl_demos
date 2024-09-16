import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Dash
from dash.dependencies import Input, Output


# instantiate a Dash object
app = Dash(__name__)


w = np.linspace(-20, 20, 100)
b = np.linspace(-20, 20, 100)

w_x, b_x = np.meshgrid(w, b)

J = w_x ** 2 + b_x ** 2

def surface_plot(w_contour=False, b_contour=False, j_contour=False):
  #  https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Surface.html
  # https://plotly.com/python-api-reference/generated/plotly.graph_objects.surface.contours.html
  # https://plotly.com/python/v3/3d-hover/
  # https://plotly.com/python/v3/3d-wireframe-plots/
  # https://laid-back-scientist.com/en/plotly-wireframe
    fig = go.Figure(go.Surface(x=w, y=b, z=J, colorscale="Spectral_r", showscale=False,
                              opacity=0.6, 
                              hovertemplate="J(w,b) = %{z}<br>w = %{x}<br>b = %{y}<extra></extra>",
                              contours=go.surface.Contours(
                                  x=go.surface.contours.X(highlight=w_contour, highlightcolor='#0066FF'),
                                  y=go.surface.contours.Y(highlight=b_contour, highlightcolor='#0066FF'),
                                  z=go.surface.contours.Z(highlight=j_contour, highlightcolor='#0066FF'))
                    ))

    fig.update_layout(xaxis_title="w", yaxis_title="b",
                      font=dict(family="Courier New, monospace", size=12),
                      scene=dict(xaxis = dict(title='w', range=[-20, 20], showspikes=False), 
                                 yaxis = dict(title_text='b', range=[-20, 20], showspikes=False),
                                 zaxis = dict(title_text='J(w,b)', range=[0, 800])),
                      hoverlabel=dict(bgcolor="white", font_size=16),
                      autosize=False, width=800, height=800,
                      margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
    
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], 
                               marker=dict(symbol='x', size=4, color='midnightblue', opacity=0.4)))
    
    return fig

fig = surface_plot()

app.layout = html.Div([
    dcc.Checklist(id="options", 
                  options=[{'label': 'Horizontal slice', 'value': 'j_contour'},
                           {'label': 'Vertical slice along w axis', 'value': 'w_contour'},
                           {'label': 'Vertical slice along b axis', 'value': 'b_contour'},
                           ], value=[], labelStyle= {"margin": "0.8rem"}, style = {'display': 'flex'}),
    dcc.Graph(id="plot"),
    html.Div(id="output")
    ])
# https://plotly.com/python/3d-camera-controls/

 
@app.callback(Output("plot", 'figure'),
              Input(component_id='options', component_property='value'),
              Input("plot", "hoverData"),
              Input("plot", "relayoutData")
              )
def create_graph(options, point, camera_info):  
    config = ['w_contour', 'b_contour', "j_contour"]
    arg = [True if option in options else False for option in config]
    fig = surface_plot(*arg)
    if camera_info:
        camera = camera_info['scene.camera']
        fig.update_layout(scene_camera=camera) 
    
    if point:
         data = point['points'][0]
         fig.add_trace(go.Scatter3d(x=[data['x'], data['x']], y=[data['y'], data['y']], z=[data['z'], 0], 
                                    hovertemplate="",
                                    marker=dict(symbol='x', size=3, color='red', opacity=0.4)))
         
    return fig

app.run(debug=True)