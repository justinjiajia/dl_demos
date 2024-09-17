import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


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

             #GithubIcon {visibility: hidden;
             }   
   
</style>
"""

st.markdown(css_str, unsafe_allow_html= True)

@st.cache_data()
def load_data(url):  
    return pd.read_csv(url, names=["size", "no. of bedrooms", "price"])


housing = load_data('https://raw.githubusercontent.com/justinjiajia/datafiles/main/portland_housing_price_2f.csv')


x, y = housing['size'].values/1000, housing['price'].values/1000

 
lm = LinearRegression()
lm.fit(x.reshape(-1, 1), y)
print(lm.intercept_, lm.coef_[0])

@st.cache_data
def fit_model(x, y):
    lm = LinearRegression()
    lm.fit(x.reshape(-1, 1), y)
    minimum_cost = compute_cost(x, y, lm.coef_[0], lm.intercept_)
    return lm.intercept_, lm.coef_[0], minimum_cost


@st.cache_data
def compute_cost(x, y, slope, intercept):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    total_cost = ((y - intercept - slope * x)**2).sum()/ (2*len(x))

    return total_cost


@st.cache_data
def data_for_line_plot(x):
    x_eval = np.linspace(min(x)-0.5, max(x)+0.5, 30)
    return x_eval

x_eval = data_for_line_plot(x)



col1, col2, _ = st.columns(3)


intercept = col1.slider(r"Choose $b$", 0.0, 150.0)
slope = col2.slider(r"Choose $w$", 0.0, 300.0)


st.markdown(r"Linear regression model:  $y ="+ f"{slope:.2f}" + r" x" +  
             f"+ {intercept:.2f}" + r"$",
            unsafe_allow_html=True)

fig = plt.figure(figsize=(4.5, 3))
ax = fig.gca()

ax.scatter(x, y, s=30, c='r', alpha=0.6, linewidths=0) 
ax.plot(x_eval, intercept + slope * x_eval, c=(66/255, 133/255, 244/255, 1), linewidth=1.5)    

ax.set(xlim=(0.5, 5), ylim=(-30, 720))
ax.set_xlabel('x', fontsize=8, fontfamily="monospace")
ax.set_ylabel('y', fontsize=8, fontfamily="monospace")
ax.tick_params(axis='both', which='major', labelsize=6, labelfontfamily="monospace")
ax.vlines(x, intercept + slope * x, y, alpha=0.3, color="k", lw=1)  

ax.text(0.65, 640, r"$J(w,b) = $" + f"{compute_cost(x, y, slope, intercept):.2f}", c='k', 
        fontsize=8, fontfamily="monospace")



st.pyplot(fig, use_container_width=False)

best_b, best_w, min_cost = fit_model(x, y)

st.info(r"Note: The best values for $w$ and $b$ are " + f"{best_w:.2f} and {best_b:.2f}.")