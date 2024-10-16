import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
  


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
    [data-testid="stMarkdownContainer"] p {
       text-align: center;
    }
      
</style>
"""

st.markdown(css_str, unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 0.6])

no_of_nodes = col1.slider(r"Choose no. of nodes per layer", 50, 500, 100, step=50)
sd = col1.slider(r"Choose standard deviation", 0.02, 0.2, 0.02, step=0.02)

strategy = col2.selectbox("Choose strategy", ["Naive initialization", "He initialization"])

D = np.random.randn(1000, 500)      # 1000 examples, each with 500 features
hidden_layer_sizes = [no_of_nodes] * 10
mpl.rcParams['font.family'] = ['monospace']

act = {'relu': lambda x: np.maximum(0, x), 'tanh': lambda x: np.tanh(x)}
Hs = {}

nonlinearities =  ['relu'] * len(hidden_layer_sizes)


if strategy == 'Naive initialization':

    st.markdown(r"Weights drawn randomly from $\mathcal{N} \sim" + f"(0, {sd})$")

else:
    st.markdown(r"Weights drawn randomly from $\mathcal{N} \sim (0,   \frac{1}{\sqrt{n^{[l]} /2}})$")

for i in range(len(hidden_layer_sizes)):
  X = D if i == 0 else Hs[i-1]
  fan_in = X.shape[1]
  fan_out = hidden_layer_sizes[i]
  if strategy == 'Naive initialization':
     W = np.random.randn(fan_in, fan_out) * sd
  else:
     W = np.random.randn(fan_in, fan_out) * (np.sqrt(2/fan_in))   
  H = np.dot(X, W)
  H =  act[nonlinearities[i]](H)
  Hs[i] = H

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for ax, (i, H) in zip(axes.ravel(), Hs.items()):
    histogram = ax.hist(H.ravel(), 100, range=(-1, 1), color=(66/255, 133/255, 244/255, 1))

    ax.set_title(f"Layer {i+1}", fontsize=8, pad=4)
    ax.text(-0.98, histogram[0].max()*0.95, f"mean = {H.mean():.2f}", c='k', fontsize=8)
    ax.text(-0.98, histogram[0].max()*0.85, f"std = {H.std():.2f}", c='k', fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=6, labelcolor="grey",
                  color="grey", length=2, width=0.5, pad=0.5)
    # ax.set_xlabel("Size", fontsize=14, labelpad=12)
    # ax.set_ylabel("No. of\n Bedrooms", fontsize=14, labelpad=12)


plt.tight_layout()
st.pyplot(fig, use_container_width=False)