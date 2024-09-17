import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout='wide')

css_str = """
<style>
             label[data-testid="stWidgetLabel"] p, [data-testid="stNotificationContentInfo"] p, [data-testid="stMarkdownContainer"] p{
                 font-size: 20px;
                 font-family: system-ui; 

             }   
             [data-testid="stElementToolbar"] {
                    display: none;
                }    
             [data-testid="stToolbar"] {
                visibility: hidden;
             }   
      
</style>
"""

st.markdown(css_str, unsafe_allow_html= True)

st.header("Training data")


@st.cache_data
def create_data():

   return pd.concat([pd.DataFrame({'size': [6.1, 6.2, 7.2, 7.3, 8, 3, 4, 4, 4.5, 5, 5.1, 4.9],
              'age': [19, 33, 29, 21, 40, 22, 29, 37, 23, 31, 26, 39],
              'malignancy': np.zeros(12)}), 
              pd.DataFrame({'size': [6, 6, 7, 7.3, 7.3, 7.8, 8.2, 8.1, 8.5, 9, 9, 9, 9.5],
              'age': [47, 28, 44, 39, 50, 35, 46, 54, 34, 30, 39, 50, 43],
              'malignancy': np.ones(13)})], axis=0)

df = create_data()

st.dataframe(df)

st.divider()

def fit_model(df):
    model = LogisticRegression(penalty=None)
    model.fit(df.iloc[:, :2], df['malignancy'].values)
    return model.intercept_, model.coef_

intercept, slope = fit_model(df)

@st.cache_data
def data_for_decision_boundary():
    x_eval = np.linspace(2, 10, 30)
    y_eval = np.linspace(15, 60, 30)
    return x_eval, y_eval

x_eval, y_eval = data_for_decision_boundary()

col1, *_ = st.columns(3)
col2, col3, _ = st.columns(3)

b = col1.slider(r"Choose $b$", -20.0, 20.0, intercept[0])
w1 = col2.slider(r"Choose $w_1$", 0.1, 3.0, slope[0, 0])
w2 = col3.slider(r"Choose $w_2$", -0.3, 0.3, slope[0, 1])

# st.markdown(r"Logistic regression model:  $\frac{1}{1+e^{-(" f"{w1:.2f}"+r"\times x_1 + " 
#            + f"{w2:.2f}" + r"\times x_2 +" + f"{b:.2f}" + r")}}$")

st.markdown(r"Logistic regression model:  $\sigma(z) = \frac{1}{1+e^{-z}}$ where $z=" + 
            f"{w1:.2f}" + r" x_1" +  
             f"{"-" if w2 < 0 else "+"} {abs(w2):.2f}"+ r" x_2" + 
             f"{"-" if b < 0 else "+"} {abs(b):.2f}" + r"$",
            unsafe_allow_html=True)

fig, ax= plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(df.loc[df['malignancy']==1, 'size'], df.loc[df['malignancy']==1, 'age'],  marker='x', s=30, color="#E94235", lw=1)
ax.scatter(df.loc[df['malignancy']==0, 'size'], df.loc[df['malignancy']==0, 'age'], s=30,
           edgecolor="#6699FF", facecolor='w', lw=1)

if w2 == 0.0:
    ax.vlines(-b / w1, ymin=15, ymax=60, color='#FF40FF', lw=1.2) 
else:    
    ax.plot(x_eval , (-b - w1 * x_eval)/ w2, c='#FF40FF', lw=1.2)



ax.set( ylim=[15, 60], xlim=(2, 10))
ax.tick_params(axis='both', which='major', labelsize=4, labelfontfamily="monospace")
ax.set_xlabel('Size', fontsize=6, fontfamily="monospace")
ax.set_ylabel('Age', fontsize=6, fontfamily="monospace")

col4, col5 = st.columns(2)

col4.pyplot(fig, use_container_width=False)


st.info(r"Note: The best values for $w_1$, $w_2$, and $b$ are " + f"{slope[0, 0]:.2f}, {slope[0, 1]:.2f}, and {intercept[0]:.2f}.")


def sigmoid(x):
  return 1/(1+np.exp(x)**(-1))


@st.cache_data
def data_for_function_graph():

    return np.meshgrid(np.linspace(2, 10, 500), np.linspace(15, 60, 500))

xx1, xx2 = data_for_function_graph()

log_surf = sigmoid(xx1 * w1 + xx2* w2 + b)

fig_3d = go.Figure(go.Surface(x=xx1, y=xx2, z=log_surf, colorscale=["Silver", "Silver"], showscale=False,
                              opacity=0.35, 
                              hovertemplate="Prediction = %{z}<br>x1 = %{x}<br>x2 = %{y}<extra></extra>",
                              contours=go.surface.Contours(
                                  x=go.surface.contours.X(highlight=False),
                                  y=go.surface.contours.Y(highlight=False),
                                  z=go.surface.contours.Z(highlight=False))
                    ))

fig_3d.add_trace(go.Scatter3d(x=df.loc[df['malignancy']==1, 'size'], y=df.loc[df['malignancy']==1, 'age'], z=np.ones(13),
                               mode="markers",  hovertemplate="",
                               marker=dict(symbol='x', size=4, color="#E94235",  line_width=0)))

fig_3d.add_trace(go.Scatter3d(x=df.loc[df['malignancy']==0, 'size'], y=df.loc[df['malignancy']==0, 'age'], z=np.zeros(12), 
                              mode="markers",  hovertemplate="",
                               marker=dict(symbol='circle-open', size=10, 
                                           color="#6699FF", line=dict(color="#6699FF", width=32))
                )
)



if w2 == 0.0:
    xx_ends = np.array([-b/w1, -b/w1])
    xy_ends = np.array([15, 60])
else: 
    xx_ends = np.array([x_eval[0],  x_eval[-1]])
    xy_ends = (-b - w1 * xx_ends)/ w2

yy_ends = np.array([y_eval[0],  y_eval[-1]])
yx_ends = (-b - w2 * yy_ends)/ w1

# https://stackoverflow.com/questions/77265426/python-plotly-planes-perpendicular-to-axes-in-3d-plot
# the vertical plane
fig_3d.add_trace(go.Mesh3d(x=np.repeat(xx_ends, 2), y=np.repeat(xy_ends, 2), z=np.array([0, 1] * 2), 
                           i=[0, 1],
                           j=[1, 3],
                           k=[2, 2], color='#FF40FF', opacity=0.1, hovertemplate=""))

# the two horizontal planes

#fig_3d.add_trace(go.Mesh3d(x=np.array([yx_ends[1], yx_ends[1], yx_ends[0], 2, 2]), 
#                           y=np.array([yy_ends[0], yy_ends[1], yy_ends[0], yy_ends[0], yy_ends[1]]), z=np.array([0.5] * 5),
#                           i=[0, 0, 3],
#                           j=[1, 1, 4],
#                           k=[2, 3, 1],  
#                           color="#6699FF", opacity=0.1, hovertemplate=""))   # blue-ish


#fig_3d.add_trace(go.Mesh3d(x=np.array([xx_ends[0], xx_ends[1], xx_ends[1]]), 
#                           y=np.array([xy_ends[0], xy_ends[0], xy_ends[1]]), z=np.array([0.5] * 3), 
#                        color="#E94235", opacity=0.1, hovertemplate=""))    # red-ish
 

if w2 < 0:

    fig_3d.add_trace(go.Mesh3d(x=np.array([xx_ends[0], 2, yx_ends[1]]), 
                            y=np.array([xy_ends[0], 60, yy_ends[1]]), z=np.array([0.5] * 3), 
                            color="#6699FF", opacity=0.1, hovertemplate=""))
    
    fig_3d.add_trace(go.Mesh3d(x=np.array([yx_ends[0], 10, xx_ends[1]]), 
                            y=np.array([yy_ends[0], 15, xy_ends[1]]), z=np.array([0.5] * 3), 
                            color="#E94235", opacity=0.1, hovertemplate=""))

if w2 == 0:

    fig_3d.add_trace(go.Mesh3d(x=np.array([2, xx_ends[0], xx_ends[1], 2]), 
                            y=np.array([15, xy_ends[0], xy_ends[1], 60]), z=np.array([0.5] * 4), 
                            # counter clockwise
                            i=[0, 2],
                            j=[1, 3],
                            k=[2, 0],  
                            color="#6699FF", opacity=0.1, hovertemplate=""))
    
    fig_3d.add_trace(go.Mesh3d(x=np.array([xx_ends[0], 10, 10, xx_ends[1]]), 
                            y=np.array([xy_ends[0], 15, 60, xy_ends[1]]), z=np.array([0.5] * 4), 
                            i=[0, 2],
                            j=[1, 3],
                            k=[2, 0],  
                            color="#E94235", opacity=0.1, hovertemplate=""))
    
if w2 > 0:

    fig_3d.add_trace(go.Mesh3d(x=np.array([xx_ends[0], 2, yx_ends[0]]), 
                            y=np.array([xy_ends[0], 15, yy_ends[0]]), z=np.array([0.5] * 3), 
                            color="#6699FF", opacity=0.1, hovertemplate=""))
    
    fig_3d.add_trace(go.Mesh3d(x=np.array([yx_ends[1], 10, xx_ends[1]]), 
                            y=np.array([yy_ends[1], 60, xy_ends[1]]), z=np.array([0.5] * 3), 
                            color="#E94235", opacity=0.1, hovertemplate=""))



#https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
fig_3d.add_trace(go.Scatter3d(x=xx_ends, y=xy_ends, z=[0.5, 0.5],  
                              mode="lines", line=dict(color='#FF40FF', width=8), hovertemplate=""))


camera = {'center': {'x': 0, 'y': 0, 'z': 0},
                                    'eye': {'x': 1.0719503491559637, 'y': -1.601593941148222, 'z': 0.9865694585895683},
          'up': {'x': 0, 'y': 0, 'z': 1}}

fig_3d.update_layout(font=dict(family="Courier New, monospace", size=12),
                     scene=dict(xaxis = dict(title='Size', range=(2, 10), showspikes=False), 
                                yaxis = dict(title_text='Age', range=[15, 60], showspikes=False),
                                zaxis = dict(title_text="Model prediction", range=[0, 1], showspikes=False),
                                camera=camera),
                    hoverlabel=dict(bgcolor="white", font_size=16),
                    autosize=False, width=600, height=600,
                    margin=dict(l=0, r=0, b=0, t=0), showlegend=False)

col5.plotly_chart(fig_3d)
