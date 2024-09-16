import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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
   
</style>
"""

st.markdown(css_str, unsafe_allow_html= True)


@st.cache_data()
def load_data(url):
    df = pd.read_csv(url, header=None)
    df.index = ['x', 'y']
    
    return df.T

data =  load_data("https://raw.githubusercontent.com/justinjiajia/machine_learning_refined/gh-pages/mlrefined_datasets/superlearn_datasets/unnorm_linregress_data.csv")
x = data['x'].to_numpy()
y = data['y'].to_numpy()


st.header("Visualize training data")

chart = px.scatter(data, x='x', y='y', width=800, height=600)
st.plotly_chart(chart)

st.divider()

@st.cache_data
def normalize(x):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x)
    return scaler.transform(x).ravel()
    

@st.cache_data
def fit_model(x, y):
    lm = LinearRegression()
    lm.fit(x.reshape(-1, 1), y)
    minimum_cost = np.mean((y - np.stack([np.ones_like(x), x], axis=1) @ np.array([lm.intercept_, lm.coef_[0]]))**2) / 2
    return lm.intercept_, lm.coef_[0], minimum_cost

@st.cache_data()
def gradient_descent(x, y, w, alpha, max_its):
    # gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
    # flatten the input function to more easily deal with costs that have layers of parameters
    # w packes the initial values for both intercept and slope
    # run the gradient descent loop
    weight_history = []      # container for weight history

    for _ in range(0, max_its+1):

        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval = np.mean((y - np.stack([np.ones_like(x), x], axis=1) @ w)**2) / 2
        diff_per_example = np.stack([np.ones_like(x), x], axis=1) @ w - y
        grad_eval = np.array([np.mean(diff_per_example), np.mean(diff_per_example * x)])
        weight_history.append(w)

        # take gradient descent step
        w = w - alpha*grad_eval

    return weight_history, cost_eval


@st.cache_data()
def data_for_contour_plot(x, y):
    para_m = np.array([*zip(B.ravel(), W.ravel())])
    para_m = np.expand_dims(para_m, axis=1)
    x_m = np.expand_dims(np.stack([np.ones_like(x), x], axis=1), axis=0)
    
    j_m = np.sum((np.expand_dims(y, axis=0) - np.sum(x_m * para_m, axis=2))**2, axis=1)/(2*len(x))
    J = j_m.reshape(B.shape)
    return J


norm = st.checkbox("Normalize feature")

col1, col2, _ = st.columns(3)

alpha = col1.slider(r"Learning rate $\alpha$", 0.6, 1.8)

if norm:
    no_iter = col2.slider("No. of gradient descent steps", 5, 20, step=5)
    x = normalize(x.reshape(-1, 1))
else:
    no_iter = col2.slider("No. of gradient descent steps", 30, 100, step=10)

B, W = np.meshgrid(np.linspace(-5, 15, 400), np.linspace(-5, 15, 400)) 
J = data_for_contour_plot(x, y)

intercept, slope, minimum_cost = fit_model(x, y)

weight_history, final_cost = gradient_descent(x, y, np.array([0, 0]), alpha, no_iter)

fig = plt.figure(figsize=(4, 4))
ax = fig.gca()

 # set level ridges
num_contours = 7
levelmin = min(J.flatten())
levelmax = max(J.flatten())
cut = 0.4
cutoff = (levelmax - levelmin)
levels = [levelmin + cutoff*cut**(num_contours - i) for i in range(0, num_contours+1)]
levels = [levelmin] + levels
levels = np.asarray(levels)

ax.contourf(B, W, J, levels=levels, cmap="Blues", alpha=0.6)
ax.contour(B, W, J, levels=levels, colors='k', alpha=0.3, linewidths=1)
ax.axvline(x=0, color='k', alpha=0.3, lw=1)
ax.axhline(y=0, color='k', alpha=0.3, lw=1)
ax.scatter(intercept, slope, c='#FF40FF', s=80, alpha=0.6,  marker="*", linewidths=0)

w0, w1 = zip(*weight_history)

# red dots represent trajectory
ax.scatter(w0, w1, s=16, c='r', zorder=3, alpha=0.3, linewidths=0)

# plot arrows
diff_w0, diff_w1 = np.diff(w0), np.diff(w1)
arrow_length = np.linalg.norm([*zip(diff_w0, diff_w1)], axis=1)
head_length = 0.2
alpha = (head_length - 0.35)/arrow_length + 1

ax.text(3.6, 14, f"Minimum cost: {minimum_cost:.4f}", c='k', 
        fontsize=6.5, fontfamily="monospace")

ax.text(3.6, 13.2, f"MSE at the final step: {final_cost:.4f}", c='k', 
        fontsize=6.5, fontfamily="monospace")

for i in range(len(diff_w0)):
    if arrow_length[i] > head_length:
        ax.arrow(w0[i], w1[i], diff_w0[i]*alpha[i], diff_w1[i]*alpha[i], head_width=0.2, 
                 head_length=head_length, fc='k', ec='k', alpha=0.4, linewidth=0.6, zorder=2, length_includes_head=True)

ax.set(xlim=(-5, 15),ylim=(-5, 15))
ax.set_xlabel('b', fontsize=8, fontfamily="monospace")
ax.set_ylabel('w', fontsize=8, fontfamily="monospace")
ax.tick_params(axis='both', which='major', labelsize=6, labelfontfamily="monospace")
ax.set_aspect("equal")

st.pyplot(fig, use_container_width=False)