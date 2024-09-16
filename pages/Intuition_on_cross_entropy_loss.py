import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

st.set_page_config(layout='wide')

css_str = """
<style>
             label[data-testid="stWidgetLabel"] p, [data-testid="stNotificationContentInfo"] p, [data-testid="stMarkdownContainer"] p{
                 font-size: 20px;
                 font-family: system-ui; 

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

x_eval = np.linspace(0.001, 0.999, 200)


st.divider()

colors=["#4285f4", "#db4437"]

setting = dict(color=colors, linewidth=0, width=0.2, tick_label=["Benign", "Malignant"])


col1, col2, _ = st.columns(3)
col3, _ = col1.columns(2)

 

type = col3.selectbox("Select class", ["Malignant: y = 1", r"Benign: y = 0"])
y_hat = col2.slider(r"Choose $\hat{y}$", 0.01, 0.99, 0.5)
include = st.checkbox("Include function graph")


_, col4, _ = st.columns(3)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))



if type.startswith("M"):
    col4.markdown(r"$ -\ln(\hat{y}) = " + f"{abs(-np.log(y_hat)):.4f}" + r"$<br/>  <br/>",
            unsafe_allow_html=True)
    ax1.bar([0, 1], [0, 1], **setting)


elif type.startswith("B"):
    col4.markdown(r"$ -\ln(1-\hat{y}) = " + f"{abs(-np.log(1-y_hat)):.4f}" + r"$<br/>  <br/>",
            unsafe_allow_html=True)
    ax1.bar([0, 1], [1, 0],  **setting)
    


if include:
    if  type.startswith("M"):
        ax3.plot(x_eval, -np.log(x_eval), c='k', lw=1.2, alpha=0.8)
        ax3.vlines(y_hat, -np.log(y_hat), 0,  color='#FF40FF', lw=1, linestyle="dashed")
        ax3.hlines(-np.log(y_hat), 0, y_hat,   color='#FF40FF', lw=1, linestyle="dashed")
        ax3.set_title(r"Graph of $-\ln{(x)}$", fontfamily="monospace")
    elif type.startswith("B"):
        ax3.plot(x_eval, -np.log(1-x_eval), c='k', lw=1.2, alpha=0.8)
        ax3.vlines(y_hat, -np.log(1-y_hat), 0,  color='#FF40FF', lw=1, linestyle="dashed")
        ax3.hlines(-np.log(1-y_hat), 0, y_hat,   color='#FF40FF', lw=1, linestyle="dashed")
        ax3.set_title(r"Graph of $-\ln{(1-x)}$", fontfamily="monospace")
    ax3.set(xlim=[-0.01, 1.01], ylim=[-0.1, 7])
    ax3.tick_params(axis='both', which='major', labelsize=10, labelfontfamily="monospace", colors='grey')

    for direction in ['bottom', 'top', 'right', 'left']:
            ax3.spines[direction].set_color('grey') 
else:
    ax3.axis("off")
          

ax1.bar_label(ax1.containers[0], [r"$1-y$", r"$y$"], padding=1) 

ax2.bar([0, 1], [1-y_hat, y_hat], **setting)

ax2.bar_label(ax2.containers[0], [r"$1-\hat{y}$", r"$\hat{y}$"], padding=1)

ax1.set_title("True probability", fontfamily="monospace")
ax2.set_title("Predicted probability", fontfamily="monospace")

for ax in [ax1, ax2]:
    for direction in ['bottom', 'top', 'right', 'left']:
        ax.spines[direction].set_color('grey') 
    ax.set(xlim=[-0.3, 1.3], ylim=[0, 1.1], xticks=[0, 1])
    ax.tick_params(axis='x', which='major', labelsize=12, labelfontfamily="monospace",  colors='grey')
    ax.tick_params(axis='y', which='major', labelsize=10, labelfontfamily="monospace",  colors='grey')
    ax.set_xlabel(None)
    plt.setp(ax.get_xticklabels()[0], color="#4285f4")
    plt.setp(ax.get_xticklabels()[1], color="#db4437")




plt.tight_layout()

st.pyplot(fig)
 