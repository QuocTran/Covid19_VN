import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py_offline
import cufflinks as cf
cf.go_offline()
py_offline.__PLOTLY_OFFLINE_INITIALIZED = True

import model_utils as mu

mu.DEATH_RATE = 0.36
mu.ICU_RATE = 0.78
mu.HOSPITAL_RATE = 2.18
mu.SYMPTOM_RATE = 10.2
mu.INFECT_2_HOSPITAL_TIME = 11
mu.HOSPITAL_2_ICU_TIME = 4
mu.ICU_2_DEATH_TIME = 4
mu.ICU_2_RECOVER_TIME = 7
mu.NOT_ICU_DISCHARGE_TIME = 5

st.title('C*apacity* I*ncidence* C*ontaining* T*esting* (CICT) Demo')
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
current_capacity = 2000

cap_ratio = st.sidebar.slider('Capacity', min_value=1.0, max_value=10.0)
contain_rate_slot = st.sidebar.empty()
test_rate_slot = st.sidebar.empty()
ct = st.sidebar.slider('Contact Tracing', value=0.2)
test_value = ct
contain_value = ct
contain_rate = contain_rate_slot.slider('Containing', value=contain_value)
test_rate = test_rate_slot.slider('Testing', value=test_value)
days_till_lock_down_end = st.sidebar.slider('How many days until lock down end?', value=7, min_value=1, max_value=60)
relax_date = dt.date.today()+dt.timedelta(days_till_lock_down_end)
forecast_horizon = st.sidebar.slider('Forecast Horizon', value=90, min_value=60, max_value=180)
state = 'New York'
#state = 'Washington'
daily, cumulative, model_beta = mu.get_metrics_by_state_US(state, lockdown_date='20200322',
                                                           forecast_horizon=forecast_horizon,
                                                           relax_date=relax_date, contain_rate=contain_rate,
                                                           test_rate=test_rate)

model_beta_new = np.append(model_beta, ((model_beta[1]+model_beta[2])*contain_rate+model_beta[1]*(1-contain_rate))-
                                        (model_beta[1]+model_beta[2]))
incidence = 'predicted_death'
#incidence = 'ICU'
incidence_func = mu.get_number_ICU_need
daily['Incidence'] = daily[[incidence]]

daily['Capacity'] = cap_ratio*current_capacity
fig = daily[['Incidence', 'Capacity']].iplot(asFigure=True)
#fig = go.Figure()
#     layout=go.Layout(
#         title=go.layout.Title(text="A Figure Specified By A Graph Object")
#     )
# )
y_upper = daily.upper_bound
y_lower = daily.lower_bound
#y_upper = incidence_func(daily.upper_bound.to_frame()).dropna()
#y_lower = incidence_func(daily.lower_bound.to_frame()).dropna()
#x = daily.index.to_pydatetime()
#x = daily.index[(daily.index>=min(y_upper.index))&(daily.index<=max(y_upper.index))].to_pydatetime()
x = y_upper.index.date
#x = pd.to_datetime(x).date
#y_upper.reset_index(drop=False,inplace=True)
fig.add_trace(go.Scatter(
    x=x,
    y=y_upper.values.flatten(),
    fill=None,
    showlegend=True,
    name='Upper Bound'))

fig.add_trace(go.Scatter(
    x=x,
    y=y_lower.values.flatten(),
    fill='tonexty',
    fillcolor='rgba(128,128,128,0.2)',
    showlegend=True,
    name='Lower Bound'
))
st.plotly_chart(fig)


log_fit, model_beta_log = mu.get_log_daily_predicted_death_by_state_US(state, lockdown_date='20200322',
                                                                       forecast_horizon=forecast_horizon,
                                                                       relax_date=relax_date, contain_rate=contain_rate)
st.subheader('Fitted log of incidences')
log_fit.rename(columns={'predicted_death':'Predicted_Incidence', 'death': 'Incidence'}, inplace=True)
fig = log_fit.drop(columns=['lower_bound', 'upper_bound', 'Incidence'], errors='ignore').iplot(asFigure=True)
x = log_fit.index
y_upper = log_fit.upper_bound.values
y_lower = log_fit.lower_bound.values
fig.add_trace(go.Scatter(
    x=x,
    y=y_upper,
    fill=None,
    showlegend=False,
    name='Upper Bound'))

fig.add_trace(go.Scatter(
    x=x,
    y=y_lower,
    fill='tonexty',
    fillcolor='rgba(128,128,128,0.1)',
    showlegend=False,
    name='Lower Bound'
))
st.plotly_chart(fig)


st.write(pd.concat([pd.Series(model_beta), pd.Series(model_beta_new)], axis=1))
st.write('Will be available at https://aipert.org')