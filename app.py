import streamlit as st
import datetime as dt
from pandas import date_range, to_datetime
import numpy as np
import model_utils as mu
import plotly.graph_objects as go
import plotly.offline as py_offline
import cufflinks as cf
cf.go_offline()
py_offline.__PLOTLY_OFFLINE_INITIALIZED = True

mu.DEATH_RATE = 0.36
mu.ICU_RATE = 0.78
mu.HOSPITAL_RATE = 2.18
mu.SYMPTOM_RATE = 10.2
mu.INFECT_2_HOSPITAL_TIME = 11
mu.HOSPITAL_2_ICU_TIME = 4
mu.ICU_2_DEATH_TIME = 4
mu.ICU_2_RECOVER_TIME = 7
mu.NOT_ICU_DISCHARGE_TIME = 5

st.set_page_config(
    page_title="Dự báo Covid-19",
    initial_sidebar_state="expanded",
)
st.title('Dự báo Covid-19')
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.markdown('COVID-19:  Dự báo cho 2 tháng tới. '
            ' Cần bao nhiêu giường bệnh hoặc ICU?')


def main(scope, local, local_sub_level, policy_change_dates, forecast_horizon, forecast_fun, debug_fun, metrics, show_debug,
         show_data, back_test, last_data_date, use_vaccine_data):
    data_load_state = st.text('Forecasting...')
    try:
        daily, cumulative, model_beta = forecast_fun(local, local_sub_level,
                                                     scope=scope,
                                                     forecast_horizon=forecast_horizon,
                                                     policy_change_dates=policy_change_dates,
                                                     back_test=back_test, last_data_date=last_data_date,
                                                     use_vaccine_data=use_vaccine_data)

    except ValueError as e:
        st.error('Chưa đủ số liệu về tử vong để dự báo. Kiểm tra lại thông tin đầu vào và ngày giãn cách')
        mu.append_row_2_logs([dt.datetime.today(), scope, local, local_sub_level, policy_change_dates, forecast_horizon,
                              last_data_date, e], 'logs/app_errors.log')
        return None
    except IndexError as e:
        st.error('Code có lỗi. Báo ngay cho tác giả!')
        mu.append_row_2_logs([dt.datetime.today(), scope, local, local_sub_level, policy_change_dates, forecast_horizon,
                              last_data_date, e], 'logs/app_errors.log')
        return None

    data_load_state.text('Đang dự báo.. Hoàn thành!')

    st.subheader('Số tử vong')
    show_metrics = ['death', 'predicted_death', '7d_avg_death']
    fig = daily[show_metrics].rename(columns={'death': 'công bố',
                                              'predicted_death': 'dự báo',
                                              '7d_avg_death': 'trung bình 7 ngày'})\
        .drop(columns=['ICU', 'hospital_beds'], errors='ignore').iplot(asFigure=True)
    x = daily.index
    y_upper = daily.upper_bound.values
    y_lower = daily.lower_bound.values

    fig.add_trace(go.Scatter(
        x=x,
        y=y_upper,
        fill=None,
        line_color='rgba(128,128,128,0)',
        legendgroup='Confidence Interval',
        showlegend=True,
        name='Cận trên'))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_lower,
        fill='tonexty',
        fillcolor='rgba(66, 164, 245,0.1)',
        line_color='rgba(128,128,128,0)',
        showlegend=True,
        legendgroup='Confidence Interval',
        name='Cận dưới'
    ))
    if back_test:
        max_y = np.nanmax(y_upper)
        fig.add_trace(go.Scatter(
            x=[last_data_date,last_data_date],
            y=[0, max_y],
            opacity=0.5,
            line_color='grey',
            mode='lines',
            hovertext=str(last_data_date),
            hoverinfo="x+name",
            name='Dùng dữ liệu đến ngày'
        ))
    fig.update_layout(
        title="Covid19 Số Hàng Ngày " + local + ", " + local_sub_level,
        yaxis_title="Tử vong",
        hovermode='x',
        legend_title='<b> Tử vong </b>',
    )
    st.plotly_chart(fig)

    show_metrics = ['death', 'predicted_death']
    fig = cumulative[show_metrics].rename(columns={'death': 'công bố',
                                                   'predicted_death': 'dự báo'})\
        .iplot(asFigure=True)
    x = cumulative.index
    y_upper = cumulative.upper_bound.values
    y_lower = cumulative.lower_bound.values
    fig.add_trace(go.Scatter(
        x=x,
        y=y_upper,
        fill=None,
        line_color='rgba(128,128,128,0)',
        legendgroup = 'CI',
        name='Cận trên'))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_lower,
        fill='tonexty',
        fillcolor='rgba(66, 164, 245,0.1)',
        line_color='rgba(128,128,128,0)',
        legendgroup='CI',
        name='Cận dưới'
    ))
    if back_test:
        max_y = np.nanmax(y_upper)
        fig.add_trace(go.Scatter(
            x=[last_data_date,last_data_date],
            y=[0, max_y],
            opacity=0.5,
            line_color='grey',
            mode='lines',
            hovertext=str(last_data_date),
            hoverinfo="x+name",
            name='Ngày dữ liệu cuối'
        ))

    fig.update_layout(
        title="Covid19 Tích lũy " + local + ", " + local_sub_level,
        yaxis_title="Tử vong",
        hovermode='x',
        legend_title='<b> Tử vong </b>'
    )

    st.plotly_chart(fig)

    if show_debug:
        log_fit, _ = debug_fun(local, local_sub_level, scope=scope, forecast_horizon=forecast_horizon, policy_change_dates=policy_change_dates,
                               back_test=back_test, last_data_date=last_data_date)
        fig = log_fit.rename(columns={'death': 'trung bình 7 ngày' , 'orig_death': 'công bố', 'predicted_death': 'dự báo'})\
            .drop(columns=['lower_bound', 'upper_bound', 'time_idx'], errors='ignore').iplot(asFigure=True)
        x = log_fit.index
        y_upper = log_fit.upper_bound.values
        y_lower = log_fit.lower_bound.values
        fig.add_trace(go.Scatter(
            x=x,
            y=y_upper,
            fill=None,
            line_color='rgba(128,128,128,0)',
            legendgroup='CI',
            name='Cận trên'))

        fig.add_trace(go.Scatter(
            x=x,
            y=y_lower,
            fill='tonexty',
            fillcolor='rgba(66, 164, 245,0.1)',
            line_color='rgba(128,128,128,0)',
            legendgroup='CI',
            name='Cận dưới'
        ))
        if back_test:
            max_y = np.nanmax(y_upper)
            fig.add_trace(go.Scatter(
                x=[last_data_date, last_data_date],
                y=[0, max_y],
                opacity=0.5,
                line_color='grey',
                mode='lines',
                hovertext=str(last_data_date),
                hoverinfo="x+name",
                name='Dùng dữ liệu đến ngày'
            ))
        fig.update_layout(
            title="Đường logarithm của số ca tử vong hàng ngày",
            yaxis_title="Logarithm của số ca tử vong hàng ngày",
            hovermode='x',
            legend_title='<b> Tử vong </b>',
        )
        st.plotly_chart(fig)

    st.subheader('Dự báo số ca và các nguồn lực y tế thiết yếu')
    st.markdown('Do hệ thống y tế ở các địa phương khác nhau, nếu dùng các chỉ số này để lập kế hoạch, '
                'cần kiểm tra hộp thông số nâng cao để cập nhật các thông số tương ứng. ')
    fig = daily.rename(columns={'ICU': 'Số bệnh nhân ở ICU', 'hospital_beds': 'số bệnh nhân nằm viện'})\
        .drop(columns=['lower_bound', 'upper_bound', '7d_avg_death'], errors='ignore').iplot(asFigure=True)
    x = daily.index
    y_upper = daily.upper_bound.values
    y_lower = daily.lower_bound.values
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y_upper,
    #     fill=None,
    #     line_color='rgba(128,128,128,0)',
    #     showlegend=False,
    #     name='Upper Bound'))
    #
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y_lower,
    #     fill='tonexty',
    #     fillcolor='rgba(128,128,128,0.1)',
    #     line_color='rgba(128,128,128,0)',
    #     showlegend=False,
    #     name='Lower Bound'
    # ))
    for invisible in metrics:
        fig.update_traces(
            visible='legendonly',
            selector=dict(name=invisible)
        )
    fig.update_traces(line=dict(dash='dot'))
    for observe_ln in ['death', 'confirmed']:
        fig.update_traces(
            line=dict(dash='solid'),
            selector=dict(name=observe_ln)
        )
    fig.update_traces(
        line=dict(color='teal'),
        selector=dict(name='confirmed'))
    if back_test:
        max_y = np.nanmax(y_upper)
        fig.add_trace(go.Scatter(
            x=[last_data_date,last_data_date],
            y=[0, max_y],
            opacity=0.5,
            line_color='grey',
            mode='lines',
            hovertext=str(last_data_date),
            hoverinfo="x+name",
            name='Dùng dữ liệu đến ngày'
        ))
    if scope == 'US' and local_sub_level == 'All':
        hospital_cap = mu.get_US_State_hospital_cap_data()
        try:
            fig.add_trace(go.Scatter(
                x=x,
                y=[hospital_cap.loc[local].Total_Hospital_Beds]*len(x),
                mode='lines',
                visible='legendonly',
                name='total hospital beds capacity'
            ))
            fig.add_trace(go.Scatter(
                x=x,
                y=[hospital_cap.loc[local].Total_ICU_Beds] * len(x),
                mode='lines',
                visible='legendonly',
                name='total ICU beds capacity'
            ))
        except KeyError:
            pass
    fig.update_layout(
        title="Covid19 Hàng Ngày " + local + ", " + local_sub_level,
        hovermode='x',
        legend_title='<b> Số ... </b>',
    )

    st.plotly_chart(fig)

    fig = cumulative.drop(columns=['lower_bound', 'upper_bound', '7d_avg_death'], errors='ignore').iplot(asFigure=True)
    x = cumulative.index
    y_upper = cumulative.upper_bound.values
    y_lower = cumulative.lower_bound.values
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y_upper,
    #     fill=None,
    #     line_color='rgba(128,128,128,0)',
    #     showlegend=False,
    #     name='Upper Bound'))
    #
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y_lower,
    #     fill='tonexty',
    #     fillcolor='rgba(128,128,128,0.1)',
    #     line_color='rgba(128,128,128,0)',
    #     showlegend=False,
    #     name='Lower Bound'
    # ))
    for invisible in metrics:
        fig.update_traces(
            visible='legendonly',
            selector=dict(name=invisible)
        )
    fig.update_traces(line=dict(dash='dot'))
    for observe_ln in ['death', 'confirmed']:
        fig.update_traces(
            line=dict(dash='solid'),
            selector=dict(name=observe_ln)
        )
    if back_test:
        max_y = np.nanmax(y_upper)
        fig.add_trace(go.Scatter(
            x=[last_data_date, last_data_date],
            y=[0, max_y],
            opacity=0.5,
            line_color='grey',
            mode='lines',
            hovertext=str(last_data_date),
            hoverinfo="x+name",
            name='Dùng dữ liệu đến ngày'
        ))

    fig.update_layout(
        title="Covid19 Tích lũy " + local + ", " + local_sub_level,
        hovermode='x',
        legend_title='<b> Số ... </b>',
    )

    st.plotly_chart(fig)

    if show_data:
        st.subheader('Dữ liệu xuất')
        st.markdown(mu.get_table_download_link(daily,
                    filename='daily_'+local+'_'+local_sub_level+'_'+str(dt.date.today())+'.csv'),
                    unsafe_allow_html=True)
        st.write('Hàng ngày', daily)
        st.markdown(mu.get_table_download_link(cumulative,
                    filename='cumulative_' + local + '_'+local_sub_level+'_' + str(dt.date.today()) + '.csv'),
                    unsafe_allow_html=True)
        st.write('Tích lũy', cumulative)
    mu.append_row_2_logs([dt.datetime.today(), scope, local, model_beta], 'logs/fitted_models.csv')


run_click = st.sidebar.button('Click to run')
scope = st.sidebar.selectbox('Thế giới, Mỹ hoặc Việt Nam', ['World', 'US', 'VN'], index=0)
if scope == 'World':
    #data_load_state = st.text('Loading data...')
    death_data = mu.get_data(scope='global', type='deaths')
    #data_load_state.text('Loading data... done!')
    local = st.sidebar.selectbox('Nước', death_data.Country.unique(), index=181)
    local_sub_level = st.sidebar.selectbox('Tỉnh/Thành Phố/State', ['All', ] + death_data.query('Country == "{}"'\
                                                        .format(local)).State.dropna().unique().tolist(), index=0)
    forecast_fun = mu.get_metrics_by_country
    debug_fun = mu.get_log_daily_predicted_death_by_country
    policy_date_fun = mu.get_policy_change_dates_by_country
elif scope == 'US':
    #data_load_state = st.text('Loading data...')
    death_data = mu.get_data(scope=scope, type='deaths')
    #data_load_state.text('Loading data... done!')
    local = st.sidebar.selectbox('State', death_data.State.unique(), index=5)
    local_sub_level = st.sidebar.selectbox('County', ['All', ] + death_data.query('State == "{}"'\
                        .format(local)).County.dropna().unique().tolist(), index=0)

    forecast_fun = mu.get_metrics_by_state
    debug_fun = mu.get_log_daily_predicted_death_by_state
    policy_date_fun = mu.get_policy_change_dates_by_state_US
elif scope == 'VN':
    mu.DEATH_RATE = 1.25
    mu.ICU_RATE = 3.75
    mu.HOSPITAL_RATE = 7.5
    mu.SYMPTOM_RATE = 12.5
    mu.INCUBATE_TIME = 4
    mu.INFECT_2_HOSPITAL_TIME = 11
    mu.HOSPITAL_2_ICU_TIME = 4
    mu.ICU_2_DEATH_TIME = 4
    mu.ICU_2_RECOVER_TIME = 7
    mu.NOT_ICU_DISCHARGE_TIME = 5
    death_data = mu.get_data(scope=scope, type='deaths')
    # data_load_state.text('Loading data... done!')
    local = st.sidebar.selectbox('Tỉnh/Thành Phố', death_data.State.unique(), index=0)
    local_sub_level = st.sidebar.selectbox('Quận/Huyện', ['All', ] + death_data.query('State == "{}"' \
                                                                                  .format(
        local)).County.dropna().unique().tolist(), index=0)

    forecast_fun = mu.get_metrics_by_state
    debug_fun = mu.get_log_daily_predicted_death_by_state
    policy_date_fun = mu.get_policy_change_dates_by_state_VN

default_dates = policy_date_fun(local)
default_dates = [to_datetime(pdate).date() for pdate in default_dates]
default_dates = list(filter(None, default_dates))
date_options = date_range(start='2020/02/01', end=dt.date.today()+dt.timedelta(7)).tolist()
date_options = default_dates + [s.date() for s in date_options[::-1]]
policy_change_dates = st.sidebar.multiselect('Ngày thay đổi chính sách, như giãn cách, phong tỏa..'
                                             ' RẤT QUAN TRỌNG để có dự báo chính xác',
                                             options=date_options, default=default_dates)
policy_change_dates.sort()
forecast_horizon = st.sidebar.slider('Độ dài dự báo (ngày)', value=60, min_value=30, max_value=90)
show_debug = st.sidebar.checkbox('Hiển thị đường logarithm số ca tử vong', value=True)
use_vaccine_data = st.sidebar.checkbox('Dùng dữ liệu về vắc xin trong mô hình', value=True)
if scope == 'VN':
    use_vaccine_data = False
'Bạn chọn: ', local, ', ', local_sub_level, 'với ngày thay đổi chính sách:', \
    [date_obj.strftime('%Y-%m-%d') for date_obj in policy_change_dates], \
    'Nhấn **Run** ở bên trái màn hình để xem dự báo. Đồ thị tương tác được. Chạy tốt nhất trên máy tính.'
show_data = st.sidebar.checkbox('Hiện thị dữ liệu xuất')

metrics = ['infected']
if st.sidebar.checkbox('Ẩn một số chỉ số'):
    metrics = st.sidebar.multiselect('Chỉ số nào bạn muốn ẩn?',
                                     ('death', 'predicted_death', 'infected', 'symptomatic',
                                      'hospitalized', 'confirmed', 'ICU', 'hospital_beds'),
                                     metrics)

back_test = st.sidebar.checkbox('Chạy back test để đánh giá')
last_data_date = dt.date.today()
if back_test:
    last_data_date = st.sidebar.date_input('Dùng dữ liệu đến ngày ', dt.date.today()+dt.timedelta(-14))
    'Chạy back test với dữ liệu đến ngày', last_data_date

if st.sidebar.checkbox('Nâng cao: thay đổi các giả định'):
    if st.sidebar.checkbox('Thay đổi tỉ lệ - phần trăm'):
        mu.DEATH_RATE = st.sidebar.slider('Tỉ lệ tử vong', value=mu.DEATH_RATE,
                                       min_value=0.01, max_value=10.0, step=0.01)
        mu.ICU_RATE = st.sidebar.slider('Tỉ lệ sử dụng ICU', value=max(mu.ICU_RATE, mu.DEATH_RATE),
                                       min_value=mu.DEATH_RATE, max_value=15.0, step=0.01)
        mu.HOSPITAL_RATE = st.sidebar.slider('Tỉ lệ nhập viện', value=max(mu.ICU_RATE, mu.HOSPITAL_RATE),
                                       min_value=mu.ICU_RATE, max_value=20.0, step=0.01)
        mu.SYMPTOM_RATE = st.sidebar.slider('Tỉ lệ có triệu chứng nặng', value=max(mu.SYMPTOM_RATE, mu.HOSPITAL_RATE),
                                       min_value=mu.HOSPITAL_RATE, max_value=25.0, step=0.01)
    if st.sidebar.checkbox('Thay  đổi thời gian - ngày'):
        mu.INFECT_2_HOSPITAL_TIME = st.sidebar.slider('Từ nhiễm đến nhập viện',
                                                      value=mu.INFECT_2_HOSPITAL_TIME, min_value=1, max_value=21)
        mu.HOSPITAL_2_ICU_TIME = st.sidebar.slider('Từ nhập viện đến chuyển ICU',
                                                      value=mu.HOSPITAL_2_ICU_TIME, min_value=1, max_value=21)
        mu.ICU_2_DEATH_TIME = st.sidebar.slider('Từ chuyển ICU đến tử vong ',
                                                      value=mu.ICU_2_DEATH_TIME, min_value=1, max_value=21)
        mu.ICU_2_RECOVER_TIME = st.sidebar.slider('Từ chuyển ICU đến hồi phục ',
                                                      value=mu.ICU_2_RECOVER_TIME, min_value=1, max_value=30)
        mu.NOT_ICU_DISCHARGE_TIME = st.sidebar.slider('Thời gian xuất viện nếu không vào ICU',
                                                      value=mu.NOT_ICU_DISCHARGE_TIME, min_value=1, max_value=21)

if run_click:
    main(scope, local, local_sub_level, policy_change_dates, forecast_horizon, forecast_fun, debug_fun, metrics, show_debug,
         show_data, back_test, last_data_date, use_vaccine_data)
    model_params = [dt.datetime.today(), scope, local, local_sub_level, policy_change_dates,
                    mu.DEATH_RATE, mu.ICU_RATE, mu.HOSPITAL_RATE,
                    mu.SYMPTOM_RATE, mu.INFECT_2_HOSPITAL_TIME, mu.HOSPITAL_2_ICU_TIME, mu.ICU_2_DEATH_TIME, 
                    mu.ICU_2_RECOVER_TIME, mu.NOT_ICU_DISCHARGE_TIME, back_test, last_data_date]
    mu.append_row_2_logs(model_params)
st.sidebar.subheader('Tác giả')
st.sidebar.info(
"""
Quoc Tran [LinkedIn](https://www.linkedin.com/in/quoc-tran-wml)  
Huong Huynh [LinkedIn](https://www.linkedin.com/in/huonghuynhsjsu)  
Liên hệ: ttquoc@gmail.com  
Source Code: [Gibhub](https://github.com/QuocTran/Covid19_VN)  
Data Source: [JHU](https://coronavirus.jhu.edu/map.html)
"""
)

with st.beta_expander('Về mô hình'):
    st.subheader('Các giả định')
    st.markdown('''
            Number of **DEATH** is the most accurate metric, despite [undercount]
            (https://www.nytimes.com/2020/04/10/nyregion/new-york-coronavirus-death-count.html), especially near peak.  
            It will be used to project other metrics under these [assumptions]
            (https://midasnetwork.us/covid-19/) for Covid19:  
            - The infected fatality rate ([IFR](https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/)): 
            0.36 percent     
            - Patients need ICU: 0.78 percent (critical)  
            - Patients need hospitalized: 2.18 percent (severe)  
            - Patients with symptom: 10.2 percent   
            - Time to hospitalized since infectected: 11 days (4 days incubation and 7 days from symptom to severe)  
            - Time to ICU since hospitalized: 4 days (assume only severe case needs to be hospitalized)  
            - Time to death since ICU use: 4 days  
            - Time to recover since ICU use: 7 days  
            - 5 days discharge if not in ICU or coming back from ICU  
            Average ICU time use: 5.6 (included both dead (4) and alive(7))
            Only ICU (critical) patients can develop death  
            
            [Here]
            (https://www.mercurynews.com/2020/04/11/when-coronavirus-kills-its-like-death-by-drowning-and-doctors-disagree-on-best-treatment/)
            is an account from the news.   
            In the assumptions, we mostly use the lower range from medical literature
            because we want to calculate the minimum ICU and hospital beds needed. These assumptions are not valid in
            local where resource is limited, while people die sooner and more often on ICU just because of not enough
            ICU to put people on. E.g. Iran, Italy, New York when dead cases peak. These also provide higher number
            on ICU and hospital beds needed if lots of patients dying out of hospital as the undercount link above.''')
    st.subheader('Các dự báo')
    st.markdown('''
            1. Total number of infection at large: death*278 (not too meaningful) or infected rate in population 
            (for individual and company to quantify risk of infection, for public health dept to declare herd immunity, 
            relax lock down measurements).
            This has a **20 days lag**, ie. this number is of 20 days ago. 
            Only in total lockdown setting, we can use the cummulative death from day 20th times 278 to get 
            total number of infection at large accurately. 
            Other alternative is whole population testing to get this number immediately. 
            2. With a correct forecast on number of death, we can get the forecast for number of hospital beds needed. 
            This is  used to build more hospital beds in advance.
            Each new death equal to 6 hospitalized (4+4)8 days before the death and continue for [12 days]
            (https://www.nejm.org/doi/full/10.1056/NEJMoa2002032)
            (using the 2.78% hospital rate and 0.36% death rate and 12 days average hospitalized and 
            4 days from ICU to death, 4 days from hospital to ICU).
            3. With a correct forecast number of death, we can get the forecast for number of ICU needed. 
            This is used to prepare ICU and buying ventilators or prepare for hospital white flags moment, 
            when doctors have to decide who to treat and who left to death due to constraint on ICU, ventilator. 
            This is also needed to prepare for social unrest.
            Each new death equal to 2.2 ICU beds 4 days before the death and continue for 6 days 
            (using the 0.78% ICU rate and 0.36% death rate and 5.6 days average ICU used).
            ''')
    st.subheader('Dự báo ca tử vong')
    st.markdown('''
    Since this is highly contagious disease, daily new death, which is a proxy for daily new infected cases
    is modeled as $d(t)=a*d(t-1)$ or equivalent to $d(t) = b*a^t$.   
    After a log transform, it becomes linear: $log(d(t))=logb+t*loga$ , so we can use linear regression to 
    provide forecast.   
    We actually use robust linear regressor to avoid data anomaly in death reporting.  
    There are two seperate linear curves, one before the lockdown is effective (20 days after lockdown) and one after.
    For using this prediction to infer back the other metrics (infected cases, hospital, ICU, etc..) only the before
    curve is used and valid. If we assume there is no new infection after lock down (perfect lockdown), the after
    curve only depends on the distribution of time to death since ICU. Since this is unknown, we have to fit the second
    curve. So for this piecewise linear function, we use package 
    [pwlf](https://jekel.me/piecewise_linear_fit_py/index.html#) with breakpoints set at lockdown effective date.
    
    
    WARNING: if no lock down happens, please use end date of the forecast period as lock down date. If you have info on 
    lock down date please use it to make sure the model provide accurate result.
            ''')
    st.subheader('Hệ quả')
    st.markdown('''
            Implications are observed in data everywhere:  
            
            
            0. ***Do not use only confirmed case to make decision***. It is misleading and distorted by testing capicity.
            1. Country with severe testing constraint will see death rate, case fatality rate(CFR) lurking around 10-15%
            , which is 0.36% death/2.78% hospitalized, e.g. Belgium, France, UK, Italy, Spain, Iran, .. [Link]
            (https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/). 
            While country with enough testing for all symptomatic patients see rate around 3.5% (0.36%/10.2%), e.g. 
            Germany. Country that can test most of potential patients, through contact tracing like Hong Kong,
            can get closer to 0.36%. It is very hard to get under 0.36% unless an effective cure is in hand. 
            Maybe Vietnam?   
            2. For most countries, the observed confirmed cases in the chart will be sandwich between symptomatic cases
            and hospitalized cases.
            3. After lock down, we need at least 15 days to see daily new cases peaks and around 20 days to see daily 
            new deaths peak, which is just in the most effective lock down. 
            For a less successful one, or severe limit on testing, this number of lag day is higher on new cases and 
            deaths.           
            4. The death peak is about 4 days after the cases peak, but number of confirmed cases depends on testing.   
            5. It needs about a month from the peak for new cases completely dissipate. 
            The number of death is also slow down but have a fat tail and about 20 days longer than the cases tail.            
            6. The above does not apply to country using widespread testing in place of SIP/lockdown like South Korea.            
            7. When no ICU or ventilator available, death rate can increase at most 2.2 times.
            ''')
    st.subheader('TODO')
    st.markdown('''
            1. Need to understand how long since infection, patient is no longer a source of infection to forecast
            curve after lock down period relaxed.          
            2. Upgrade the calculation using mean to use distribution if enough data is available.
            3. Use SEIR model when number of infected cases near 20% of population.''')
with st.beta_expander('Medical myths'):
    st.markdown('I am not a medical doctor. I am a statistician but I strongly believe in this:')
    st.subheader('How Vietnamese doctors treat SARS before and COVID19 now?')
    st.markdown('''
    The doctors open all doors and windows to let sunlight in and sick air out. Patients with mild symptom are suggested
    to [gargling with salt water.]
    (https://www.webmd.com/cold-and-flu/features/does-gargling-wlth-salt-water-ease-a-sore-throat#1) 
    Some rare severe patients need treatment, ventilator and/or ECMO anyway. We have no other secrete weapon beside the
    hot humid weather. By doing that, we get very low death rate, zero for now, and low infection rate for doctors. Why
    does it work? We think utmostly, this is an ***airborne*** disease. People got infected or death because of the air 
    they breath in, or more precisely by how much viruses in the air they put into their lungs at that particular 
    moment. That is why we see a mild progression, when patient's lung adapts to virus in the air, might suddenly turn to
    a [death end]
    (https://www.mercurynews.com/2020/04/11/when-coronavirus-kills-its-like-death-by-drowning-and-doctors-disagree-on-best-treatment/)
    .''')
    st.subheader('How does patient die from COVID-19?')
    st.markdown('''
    Short answer: mostly not because of the virus but because of their own immune system, particularly the cytokine 
    release syndrome. 
    
    Long answer: When patient got infected, virus can be everywhere from their respiratory tracts, blood, stool,.. even
    brain, but the critical center is in their tonsils. When the density of virus is enough, around 5-14 days after 
    infection, patients start to get a lot of dry cough, which release the virus to the air and to their lungs. 
    And then the lungs cells die very fast due to the immune system rush in and kill everything, the viruses and the
    lung cells. Hence, the disease is named severe acute respiratory syndrome (SARS). How to prevent the death happen?
    Keep patient's lung away from the viruses in the air. And as soon as you can.''')
    st.subheader('Why ventilator is so important? or not?')
    st.markdown('''
    - When lots of lung cells die, the air sacs of the lungs become filled with a gummy yellow fluid, 
    other organs got cut off from the oxygen in blood, and dying too.
    Among those who are critically ill, profound acute hypoxemic respiratory failure from ARDS is the dominant finding.
    The need for mechanical ventilation in those who are critically ill is high ranging from 42 to 100 percent. [Link]
    (https://www.uptodate.com/contents/coronavirus-disease-2019-covid-19-critical-care-issues)  
    - But the ventilator also serves a very important purpose: keep the virus away from the air patient breathes in. So if 
    your local does not have any ventilator left, you can make a cheap breathing device that can filter virus from the 
    air intake and fan away air out of patient. Like open door with big fan. Or fancier, using the [helmet]
    (https://www.nbcnews.com/news/us-news/texas-mom-pop-business-flooded-orders-helmet-ventilators-amid-coronavirus-n1173466)
    , which is proved working through [medical trial]
    (https://www.uchicagomedicine.org/forefront/patient-care-articles/helmet-based-ventilation-is-superior-to-face-mask-for-patients-with-respiratory-distress)
    . For patient that can not breath on their own, something like [this]
    (https://newatlas.com/medical/british-engineers-modern-iron-lung-covid-19-ventilator-alternative/)
    will help. Try to not let patient bubble in their own viruses, just as the patients died in the 
    [adult care facility]
    (https://abc7news.com/health/6-dead-53-infected-with-covid-19-at-hayward-assisted-living-facility/6088815/)''')
    st.subheader('How to be in the 95 percent patients that are not in critical condition?')
    st.markdown('''
    When you are positive for COVID19 but not have enough symptoms to be admitted to hospital, keep your place well 
    ventilated (open windows or fan out), well lit with sun light, and warmer than normal. 
    Gargle with salt water at least 3 times a day or when
    you want to cough. Try to cough as less as you can using any method you know, or go outside if you need to. And then
    watch your vitals, temperature and blood oxygen level for sign that you can be admitted to hospital for proper care.
    ''')
    st.subheader('How to prevent the transmission?')
    st.markdown('''
    This website shows that lockdown is actually working. So follow the lock down guideline, 
    particularly avoid [close air spaces](https://wwwnc.cdc.gov/eid/article/26/7/20-0764_article) 
    with lots of people, such as airplane, subway, church, air conditioning restaurant, etc.., 
    and ***wear mask***.''')
    st.markdown('***Please spread the message and stay safe!***')
    mu.append_row_2_logs([dt.datetime.today(), ], log_file='logs/medical_myths_logs.csv')

with st.beta_expander('Mục lục tham khảo'):
    st.markdown('[IHME COVID-19 Infection Spread](https://covid19.healthdata.org) '
                'Reason we speed up our development. Lots of thing to like. One thing '
                'we would do differently, the forecasting model.')
    st.markdown('https://www.streamlit.io Fast prototype.')
    st.markdown('[pwlf](https://jekel.me/piecewise_linear_fit_py/index.html) Key tool for my model.')
    st.markdown('https://www.uptodate.com/contents/coronavirus-disease-2019-covid-19')
    st.markdown('https://midasnetwork.us/covid-19/')
    st.markdown('[Letter from the frontline, Italy]'
                '(https://www.atsjournals.org/doi/pdf/10.1164/rccm.202003-0817LE)')
    st.markdown('[COVID-19 Hospital Impact Model for Epidemics (CHIME)](https://penn-chime.phl.io)')
    st.markdown('[UW-Madison AFI DSI Covid-19 Research Portal](https://datascience.wisc.edu/covid19/)')
    st.markdown('[Fast fact on US hospital beds]'
                '(https://www.aha.org/statistics/fast-facts-us-hospitals)')
    st.markdown('[Number of ventilators] '
                '(https://www.centerforhealthsecurity.org/resources/COVID-19/COVID-19-fact-sheets/200214-VentilatorAvailability-factsheet.pdf)')
    st.markdown('[Universal Screening for SARS-CoV-2](https://www.nejm.org/doi/10.1056/NEJMc2009316)')
    st.markdown('[All the cool charts](https://ourworldindata.org/coronavirus)')
    st.markdown('[Global Covid-19 Case Fatality Rates from Oxford]'
                '(https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/)')
    st.markdown('[Lancet: estimate the severity of coronavirus disease 2019: a model based analysis]'
                '(https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext)')
    st.markdown('2020/04/27 [Clinical Characteristics of Coronavirus Disease 2019 in China]'
                '(https://www.nejm.org/doi/full/10.1056/NEJMoa2002032)')
    st.markdown('2020/04/27 [COVID-19 Hospitalization Tracking Project from UMN]'
                '(https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project) used to validate ICU and hospital '
                'bed projection.')
    st.markdown('2020/04/27 [Correlation between HVAC used and infected rates, from a private conversation]'
                '(https://drive.google.com/drive/folders/1bLWEX8o7LQoLzwegbYpkIJUVNWS_DKlO)')
    st.markdown('2020/04/27 [Coronavirus Aerosolized Through Talking]'
                '(https://www.coronavirustoday.com/sars-cov-2-viral-particles-cause-covid-19-disease)')
    st.markdown('2020/04/27 [Research Letter from CDC: COVID-19 Outbreak Associated with Air Conditioning in Restaurant]'
                '(https://wwwnc.cdc.gov/eid/article/26/7/20-0764_article)')
    st.markdown('2020/05/04 [Great resources on data](http://covid19.laiyaconsulting.com/source)')
    st.markdown('2020/05/06 [Country lock down dates]'
                '(https://en.wikipedia.org/wiki/Curfews_and_lockdowns_related_to_the_COVID-19_pandemic)')
    st.markdown('2020/06/29 [Harvard Global Health Institute US Hospital Capacity]'
                '(https://globalepidemics.org/hospital-capacity/)')
    st.markdown('2020/06/29 [Covid19 US policy changes Github]'
                '(https://github.com/COVID19StatePolicy/SocialDistancing/)')
    st.markdown('2020/06/29 [CDC Covid19 US Death forecast hub]'
                '(https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html)')
    st.markdown('2020/06/29 [JAMA Article: Nasal ACE2 Levels and COVID-19 in Children]'
                '(https://jamanetwork.com/journals/jama/fullarticle/2766522)')
    st.markdown('2020/06/29 [CDC Public Face Mask Recommendation]'
                '(https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/cloth-face-cover.html)')
    st.markdown('2020/09/05 [Wiki National responses to the COVID-19 pandemic]'
                '(https://en.wikipedia.org/wiki/National_responses_to_the_COVID-19_pandemic)')
    st.markdown('2021/05/26 [Our World in Data Coronavirus (COVID-19) Vaccinations]'
                '(https://ourworldindata.org/covid-vaccinations)')


    st.subheader('Tin tức')
    st.markdown('https://www.mercurynews.com/2020/04/11/when-coronavirus-kills-its-like-death-by-drowning-and-doctors-disagree-on-best-treatment/')
    st.markdown('https://www.statnews.com/2020/04/08/doctors-say-ventilators-overused-for-covid-19/')
    st.markdown('https://www.uchicagomedicine.org/forefront/patient-care-articles/helmet-based-ventilation-is-superior-to-face-mask-for-patients-with-respiratory-distress')
    st.markdown('https://www.cnn.com/2020/04/17/asia/china-wuhan-coronavirus-death-toll-intl-hnk/index.html')
    st.markdown('I never believe that some days I see this long sad story in [New Yorker]'
                '(https://www.newyorker.com/news/our-local-correspondents/the-body-collectors-of-the-coronavirus-pandemic)'
                )
    st.markdown('2020/04/27 [Coronavirus lingers in air of crowded spaces, new study finds]'
                '(https://www.mercurynews.com/2020/04/27/coronavirus-lingers-in-air-of-crowded-spaces-new-study-finds/)')
    st.markdown('2020/04/27 [Air conditioning appears to spread coronavirus—but opening windows could stop it, studies suggest]'
                '(https://www.msn.com/en-xl/health/coronavirus/air-conditioning-appears-to-spread-coronavirus—but-opening-windows-could-stop-it-studies-suggest/ar-BB12GeD1)')


with st.beta_expander('Changelog'):
    st.markdown('2020/04/22 Big change on the default parameters about rates using the New York study, which suggests '
                'asymptomatic rate is 88 percent and death rate is 1 percent. This is now in line with the Chinese study'
                ' . We only need to divide every rate in Chinese study by 2.3. So hospitalized reduced to 6 pct and ICU'
                ' rate to 2.2 pct.')
    st.markdown('2020/04/27 Further adjust the default parameters about rates using Oxford estimate for IFR, which is '
                'now at 0.36%. All the other rates are changed accordingly. One extra bonus, our estimates on ICU and '
                'hospital bed are in the same range with observed numbers now. ')
    st.markdown('2020/05/07 Added back test to visually evaluate model sensitivity to new data point')
    st.markdown('2020/06/11 Added lock down end date which allows modeling the relax phase')
    st.markdown('2020/06/29 Added total hospital beds and ICU from Harvard data source')
    st.markdown('2020/08/19 Change model and UI to allow more than 2 significant policy change dates such as lock down '
                'relax')
    st.markdown('2020/08/24 Upgraded Streamlit to version 0.65 Added FB Comments Plugin')
    st.markdown('2020/12/08 Added forecast for US counties, UK, Canada, and Australia provinces. '
                'Added Disqus Comment box')
    st.markdown('2021/01/05 Using 7 days moving average and blank out 7 days after policy effective date')
    st.markdown('2021/01/19 Showed 7 days moving average on plots. Move Run button to top')
    st.markdown('2021/05/26 Added effect for vaccine and immunity from past covid infection')

disqus_js = """
<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
this.page.url = 'https://covid19.aipert.org'
this.page.identifier = my_page; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://covid19aipert.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
"""
#st.components.v1.html(disqus_js)
google_analytics_js = """
<!-- Global site tag (gtag.js) - Google Analytics -->
    <script
      async
      src="https://www.googletagmanager.com/gtag/js?id=UA-168384497-1"
    ></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag() {
        dataLayer.push(arguments);
      }
      gtag("js", new Date());

      gtag("config", "UA-168384497-1");
    </script>
    """
#st.components.v1.html(google_analytics_js)
fb_comments = """
        <div class="fb-comments" data-href="https://covid19.aipert.org" data-numposts="5" data-width=""></div>
        """
#st.components.v1.html(fb_comments)
st.components.v1.iframe('https://covid19.aipert.org/google_analytics.html', height=1, scrolling=False)
with st.beta_expander('Hiện bình luận', expanded=True):
    st.components.v1.iframe('https://covid19.aipert.org/discuss.html', height=400, scrolling=True)
    st.components.v1.iframe('https://covid19.aipert.org/disqus.html', height=400, scrolling=True)
