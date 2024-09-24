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


type_col, _, col, _ = st.columns([0.25, 0.05, 0.6, 0.1])

type = type_col.selectbox(r"Select activation function $h(z)$", ["relu", "sigmoid", "linear"])

col_1, col_2, col_3, col_4 = col.columns(4)

l2_w_1 = col_1.slider(r"Select $w_1^{[2]}$", -2.0, 2.0,  -0.8 if type == "sigmoid" else -1.3 )
l2_w_2 = col_2.slider(r"Select $w_2^{[2]}$", -2.0, 2.0, 0.6 if type == "sigmoid" else 1.3)
l2_w_3 = col_3.slider(r"Select $w_3^{[2]}$", -2.0, 2.0, 0.66)
l2_b = col_4.slider(r"Select $b^{[2]}$", -2.0, 2.0, -0.23)

# plt.style.use('fivethirtyeight')

def activation(z, type):
   if type == "sigmoid":
      return 1.0 /(1.0 + np.exp(-z * 5 )) # scale up
   elif type == "relu":
      return z * (z > 0.0)
   else:
      return z

_, text_col, _ = st.columns([0.15, 0.55, 0.3])

txt1, txt2, txt3 = text_col.columns(3)

txt1.markdown("Weighted sums")
txt2.markdown("Activations")
txt3.markdown("Contributions to output")

col1, col2, _ = st.columns([0.15, 0.55, 0.3])

col3, col4, agg_col = st.columns([0.15, 0.55, 0.3])

col5, col6, _ = st.columns([0.15, 0.55, 0.3])

l1_w_1 = col1.slider(r"Select $w_{11}^{[1]}$", -2.0, 2.0, 0.4)
l1_b_1 = col1.slider(r"Select $b_1^{[1]}$", -2.0, 2.0, -0.2)

l1_w_2 = col3.slider(r"Select $w_{21}^{[1]}$", -2.0, 2.0, 0.9)
l1_b_2 = col3.slider(r"Select $b_2^{[1]}$", -2.0, 2.0, -0.9)


l1_w_3 = col5.slider(r"Select $w_{31}^{[1]}$", -2.0, 2.0, -0.7)
l1_b_3 = col5.slider(r"Select $b_3^{[1]}$", -2.0, 2.0, 1.1)


x_eval = np.linspace(0, 2, 200)

fig1, ax1 = plt.subplots(1, 3, figsize=(4.2, 1.4),  )
ax1[0].plot(x_eval, l1_w_1 * x_eval+ l1_b_1 , c="#E94235", lw=1)
ax1[0].text(0.25, -0.85, r"$z_{1}^{[1]}= w_{11}^{[1]}x_1 + b_1^{[1]}$", 
            fontsize=7, fontfamily="monospace")

ax1[1].plot(x_eval, activation(l1_w_1 * x_eval+ l1_b_1, type), c="#E94235", lw=1)
ax1[1].text(0.5, -0.85,  r"$a_{1}^{[1]}= h\left(z_{1}^{[1]}\right)$",
            fontsize=7, fontfamily="monospace")
ax1[2].plot(x_eval, l2_w_1*activation(l1_w_1 * x_eval+ l1_b_1, type), c="#E94235", lw=1)
ax1[2].text(0.7, -0.85, r"$w_1^{[2]} a_{1}^{[1]}$", 
            fontsize=7, fontfamily="monospace")


fig2, ax2 = plt.subplots(1, 3, figsize=(4.2, 1.4))
ax2[0].plot(x_eval, l1_w_2 * x_eval+ l1_b_2, c="#6699FF", lw=1)
ax2[0].text(0.25, -0.85, r"$z_{2}^{[1]}= w_{21}^{[1]}x_1 + b_2^{[1]}$", 
            fontsize=7, fontfamily="monospace")

ax2[1].plot(x_eval, activation(l1_w_2 * x_eval+ l1_b_2, type), c="#6699FF", lw=1)
ax2[1].text(0.5, -0.85, r"$a_{2}^{[1]}= h\left(z_{2}^{[1]}\right)$", 
            fontsize=7, fontfamily="monospace")
ax2[2].plot(x_eval, l2_w_2*activation(l1_w_2 * x_eval+ l1_b_2, type), c="#6699FF", lw=1)
ax2[2].text(0.7, -0.85, r"$w_2^{[2]} a_{2}^{[1]}$", 
            fontsize=7, fontfamily="monospace")

fig3, ax3 = plt.subplots(1, 3, figsize=(4.2, 1.4))
ax3[0].plot(x_eval, l1_w_3 * x_eval+ l1_b_3, c="#F4B400" , lw=1)
ax3[0].text(0.25, -0.85, r"$z_{3}^{[1]}= w_{31}^{[1]}x_1 + b_3^{[1]}$",  
            fontsize=7, fontfamily="monospace")

ax3[1].plot(x_eval, activation(l1_w_3 * x_eval+ l1_b_3, type), c="#F4B400" ,lw=1)
ax3[1].text(0.5, -0.85, r"$a_{3}^{[1]}= h\left(z_{3}^{[1]}\right)$", 
            fontsize=7, fontfamily="monospace")

ax3[2].plot(x_eval, l2_w_3*activation(l1_w_3 * x_eval+ l1_b_3, type), c="#F4B400" ,lw=1)
ax3[2].text(0.7, -0.85, r"$w_3^{[2]} a_{3}^{[1]}$",  
            fontsize=7, fontfamily="monospace")

for fig in [fig1, fig2, fig3]:
    fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9,
                        wspace=0.1, 
                        hspace=0)

agg_fig, agg_ax = plt.subplots(1, 1, figsize=(1.2, 1.2))

agg_output = (l2_w_1*activation(l1_w_1 * x_eval+ l1_b_1, type) + 
              l2_w_2*activation(l1_w_2 * x_eval+ l1_b_2, type) + l2_w_3*activation(l1_w_3 * x_eval+ l1_b_3, type) + l2_b)

agg_ax.plot(x_eval, agg_output, c='#FF40FF', lw=1)
agg_ax.set_xlabel(r"$w_1^{[2]} a_{1}^{[1]}+w_2^{[2]} a_{2}^{[1]}+w_3^{[2]} a_{3}^{[1]}+b^{[2]}$", 
                  fontsize=5, labelpad=0.5)

agg_fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9,
                        wspace=0.1, 
                        hspace=0)

for idx, ax in enumerate([*ax1, *ax2, *ax3, agg_ax]):
   for direction in ['bottom', 'top', 'right', 'left']:
        ax.spines[direction].set_color('grey') 
   ax.hlines(xmin=0, xmax=2, y=0, linestyle="dashed", lw=0.4, color="k")

   ax.set(ylim=[-1.0, 1.0], xlim=(0.0, 2.0), xticks=[0.0, 2.0], yticks=[-1.0, 1.0], aspect='equal')
   ax.xaxis.set_minor_locator(MultipleLocator(0.2))
   ax.yaxis.set_minor_locator(MultipleLocator(0.2))
   if idx % 3 != 0:
      ax.set_yticklabels([])
   ax.tick_params(axis='both', which='major', labelsize=4, labelfontfamily="monospace", labelcolor="grey", 
                  color="grey", length=2, width=0.5)
   ax.tick_params(axis='both', which='minor', color="grey", length=1.5, width=0.5)
   


col2.pyplot(fig1, use_container_width=True)
col4.pyplot(fig2, use_container_width=True)
col6.pyplot(fig3, use_container_width=True)
agg_col.pyplot(agg_fig, use_container_width=False)