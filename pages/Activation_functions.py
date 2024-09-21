import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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

st.markdown(css_str, unsafe_allow_html= True)

def activation(z, type):
   if type == "sigmoid":
      return 1.0 /(1.0 + np.exp(-z )) 
   elif type == "tanh":
      return (1-np.exp(-2* z)) /  (1+np.exp(-2* z))
   elif type == "relu":
      return z * (z > 0.0)
   elif type == "leaky_relu":
      return np.where(z > 0, z, z * 0.1)     
   else:
      return z

def derivative(z, type):
   if type == "sigmoid":
      a = activation(z, type)
      return a * (1-a)
   elif type == "tanh":
      a = activation(z, "tanh")
      return 1- a ** 2
   elif type == "relu":
      return np.where(z > 0, 1, 0) 
   elif type == "leaky_relu":
      return np.where(z > 0, 1,  0.1)     
   else:
      return z
   
plt.close("all")

plt.style.use('seaborn-v0_8-paper')
fig1, axes1 = plt.subplots(1, 4, figsize=(14.4, 3.4))


x_eval = np.linspace(-4, 4, 200)
names = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"]

axes1[0].plot(x_eval, activation(x_eval, "sigmoid"), color='#0096ff')
axes1[0].set(ylim=(-1.2, 1.2))
axes1[0].hlines(1, -4.2, 4.2, color='k', alpha=0.2, ls="dashed")
axes1[0].text(-3, 0.68, r"$f(x)=\frac{1}{1+e^{-x}}$", c='k', fontsize=16)

axes1[1].plot(x_eval, activation(x_eval, "tanh"), color='#0096ff')
axes1[1].set(ylim=(-1.2, 1.2))
axes1[1].hlines(0, -4.2, 4.2, color='k', alpha=0.2)
axes1[1].hlines([-1, 1], [-4.2, -4.2], [4.2, 4.2], color='k', alpha=0.2, ls="dashed")
axes1[1].text(-3.2, 0.68, r"$f(x)=\frac{1-e^{-2x}}{1+e^{-2x}}$", c='k', fontsize=16)


axes1[2].plot(x_eval, activation(x_eval, "relu"), color='#0096ff')
axes1[2].set(ylim=(-4.2, 4.2))
axes1[2].text(-2.5, 2.5, r"$f(x)=\max(x, 0)$", c='k', fontsize=15,  fontfamily="monospace")

axes1[3].plot(x_eval, activation(x_eval, "leaky_relu"), color='#0096ff')
axes1[3].set(ylim=(-4.2, 4.2))
axes1[3].text(-3, 2.5, r"$f(x)=\max(x, 0.1x)$" ,  fontsize=15, fontfamily="monospace")






fig2, axes2 = plt.subplots(1, 4, figsize=(14.4, 3.4))



  
axes2[0].plot(x_eval, derivative(x_eval, "sigmoid") , color='#0096ff')
axes2[0].set(ylim=(-0.6, 0.6))
axes2[0].hlines(0.25, -4.2, 4.2, color='k', alpha=0.2, ls="dashed")



axes2[1].plot(x_eval, derivative(x_eval, "tanh") , color='#0096ff')
axes2[1].set(ylim=(-1.2, 1.2))
axes2[1].hlines(1, -4.2, 4.2, color='k', alpha=0.2, ls="dashed")

axes2[2].hlines(1, -4.2, 4.2, color='k', alpha=0.2, ls="dashed")
axes2[2].plot(x_eval[x_eval<0], derivative(x_eval[x_eval<0], "relu") , color='#0096ff')
axes2[2].plot(x_eval[x_eval>0], derivative(x_eval[x_eval>0], "relu") , color='#0096ff')
axes2[2].set(ylim=(-1.2, 1.2))


axes2[3].hlines(1, -4.2, 4.2, color='k', alpha=0.2, ls="dashed")
axes2[3].plot(x_eval[x_eval<0], derivative(x_eval[x_eval<0], "leaky_relu") , color='#0096ff')
axes2[3].plot(x_eval[x_eval>0], derivative(x_eval[x_eval>0], "leaky_relu") , color='#0096ff')
axes2[3].set(ylim=(-1.2, 1.2))

for idx, (ax,  name) in enumerate(zip(axes1.flat, names)):

  ax.vlines(0, -4.2, 4.2, color='k', alpha=0.2)
  ax.hlines(0, -4.2, 4.2, color='k', alpha=0.2)
  ax.set(xlim=(-4, 4))
  ax.xaxis.set_major_locator(MultipleLocator(2.0))
  ax.xaxis.set_minor_locator(MultipleLocator(1.0))
  if idx >=2:
     ax.yaxis.set_major_locator(MultipleLocator(2.0))
     ax.yaxis.set_minor_locator(MultipleLocator(1.0))
  ax.set_title(name, fontsize=14, pad=12)
  ax.tick_params(axis='both', which='major', labelsize=7, labelfontfamily="monospace", labelcolor="grey", 
                  color="grey", length=2, width=0.5, pad=0.5)

  
for idx, ax in enumerate(axes2.flat):

  ax.vlines(0, -4.2, 4.2, color='k', alpha=0.4)
  ax.hlines(0, -4.2, 4.2, color='k', alpha=0.4)
  ax.set(xlim=(-4, 4))
  ax.xaxis.set_major_locator(MultipleLocator(2.0))
  ax.xaxis.set_minor_locator(MultipleLocator(1.0))
  ax.tick_params(axis='both', which='major', labelsize=7, labelfontfamily="monospace", labelcolor="grey", 
                  color="grey", length=2, width=0.5, pad=0.5)

for fig in [fig1, fig2]:
    fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                            right=0.9, 
                            top=0.9,
                            wspace=0.1, 
                            hspace=0)
st.pyplot(fig1)
st.pyplot(fig2)