import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import streamlit as st
import seaborn as sns

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


st.divider()

def fit_model(df):
    model = LogisticRegression(penalty=None)
    model.fit(df.iloc[:, :2], df['malignancy'].values)
    return model.intercept_, model.coef_


intercept, slope = fit_model(df)

colors=["#4285f4", "#db4437"]

setting = dict(color=colors, linewidth=0, width=0.2, tick_label=["Benign", "Malignant"])


df = df.reset_index()

@st.cache_data
def prediction(x, w1, w2, b):
    
    return 1/(1+ np.exp(-(x @ np.array([w1, w2]) + b)))
    
@st.cache_data
def data_for_decision_boundary():
    x_eval = np.linspace(1, 12, 30)
    return x_eval

x_eval = data_for_decision_boundary()

neg_examples = df.loc[df["malignancy"]==0, 'index':"age"]
pos_examples = df.loc[df["malignancy"]==1, 'index':"age"]



col1, *_ = st.columns(3)
col2, col3, _ = st.columns(3)

b = col1.slider(r"Choose $b$", -20.0, 20.0, intercept[0])
w1 = col2.slider(r"Choose $w_1$", 0.1, 3.0, slope[0, 0])
w2 = col3.slider(r"Choose $w_2$", -0.3, 0.3, slope[0, 1])


fig, ax= plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(df.loc[df['malignancy']==1, 'size'], df.loc[df['malignancy']==1, 'age'],  marker='x', s=30, color="#E94235", lw=1)
ax.scatter(df.loc[df['malignancy']==0, 'size'], df.loc[df['malignancy']==0, 'age'], s=30,
           edgecolor="#6699FF", facecolor='w', lw=1)

if w2 == 0.0:
    ax.vlines(-b / w1, ymin=15, ymax=60, color='#FF40FF', lw=1.2) 
else:    
    ax.plot(x_eval , (-b - w1 * x_eval)/ w2, c='#FF40FF', lw=1.2)

ax.set( ylim=[15, 60], xlim=(1.8, 10.4),)
ax.tick_params(axis='both', which='major', labelsize=4, labelfontfamily="monospace")
ax.set_xlabel('Size', fontsize=6, fontfamily="monospace")
ax.set_ylabel('Age', fontsize=6, fontfamily="monospace")

col1, col2 = st.columns(2)

col1.pyplot(fig, use_container_width=False)

predictions = prediction(neg_examples.loc[:, "size":"age"].values , w1, w2, b)

pred = pd.DataFrame({"label": 1, "prob": predictions, "index": neg_examples.loc[:, "index"].values})
pred_r = pd.DataFrame({"label": 0, "prob": 1-predictions, "index": neg_examples.loc[:, "index"].values})
pred_overall = pd.concat([pred_r, pred])
pred_overall

g = sns.FacetGrid(pred_overall, col="index", col_wrap=7)
g.map_dataframe(plt.bar,"label" , "prob", **dict(color=colors,  width=0.2))
g.tick_params(labelfontfamily="monospace", labelsize=12)
g.set(xlabel=None, ylabel=None, xlim=[-0.3, 1.3], ylim=[0, 1.1], xticks=[0, 1],)
g.tight_layout()
st.pyplot(g)