import numpy as np
import pandas as pd
import streamlit as st
import datetime

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.linear_model import LinearRegression as model_constructor_lr
from sklearn.tree import DecisionTreeRegressor as model_constructor_dt
from xgboost import XGBRegressor as model_constructor_xgb
from sklearn.metrics import mean_absolute_percentage_error as metric_1
from sklearn.metrics import mean_squared_error as metric_2
from sklearn.preprocessing import OneHotEncoder


@st.cache_data
def read_and_preprocess_data():
    data = pd.read_csv(f'data/bike-sharing-hourly.csv')
    # First, we make sure that 'dteday' is a datetime type.
    data['dteday'] = pd.to_datetime(data['dteday'])

    # Now, we create a new colun adding the number of hours from the 'hr' column to 'dteday'.
    data['datetime'] = data['dteday'] + pd.to_timedelta(data['hr'], unit='h')
    data.set_index('datetime', inplace=True)
    
    # Drop irrelevant columns
    data = data.drop(['dteday', 'instant'], axis=1)
    
    #Group of variables as dictionary
    target_var = ['cnt']
    ignore_vars = ['casual', 'registered']
    categorical_vars = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    numerical_vars = list(set(data.columns) - set(categorical_vars) - set(target_var) - set(ignore_vars))
    
    # Create climate_perception
    data['climate_perception'] = data['weathersit'].rolling(window=3).apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    data['climate_perception'] = data['climate_perception'].fillna(data['climate_perception'].mode()[0])
    categorical_vars.append('climate_perception')
    # Create temperature_perception
    data['temperature_perception'] = data['temp'].rolling(window=12).mean()
    data['temperature_perception'] = data['temperature_perception'].fillna(data['climate_perception'].mean())
    numerical_vars.append('temperature_perception')
    
    #Final group variables
    var_groups = {'target': target_var,
             'ignore': ignore_vars,
             'catergorical': categorical_vars,
             'numerical': numerical_vars}
    
    #We reset the index to avoid problems
    data.reset_index(inplace=True)
    data_prep = data.copy()

    #One Hot Encoding
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output = False, drop = 'first')
    ohe.fit(data_prep[categorical_vars])

    dat_ohe = pd.DataFrame(ohe.transform(data_prep[categorical_vars]))
    dat_ohe.columns = ohe.get_feature_names_out()
    data_prep = data_prep.drop(categorical_vars, axis=1)
    data_prep = pd.concat((data_prep, dat_ohe), axis=1)

    #Set the date back as index in both datasets
    data_prep.set_index('datetime', inplace=True)
    data_prep = data_prep.drop(ignore_vars, axis=1)
    data.set_index('datetime', inplace=True)
    
    #Retrieve the prepro data, the groups, the prepared data for forecasting, the OHE model
    return data, var_groups, data_prep, ohe, categorical_vars

def read_models():
    #Read models found in the notebook
    model_lr = joblib.load('linear_reg_model.joblib')
    grid_dt = joblib.load('decision_tree_model.joblib')
    model_xg = joblib.load('xgboost_model.joblib')
    return model_lr, grid_dt, model_xg

def dataframe_prediction(select_date = datetime.date.today()):
    #Create a dataframe with the variables for the model (24 hours)
    datetime = pd.date_range(start=select_date, periods=24, freq='H')

    df = pd.DataFrame(index=datetime)

    df['season'] = df.index.month.map({12: 1, 1: 1, 2: 1,  # Winter
                                      3: 2, 4: 2, 5: 2,   # Spring
                                      6: 3, 7: 3, 8: 3,   # Summer
                                      9: 4, 10: 4, 11: 4})  # Fall
    df['yr'] = df.index.year - 2011
    df['mnth'] = df.index.month
    df['hr'] = df.index.hour
    df['holiday'] = None
    df['weekday'] = df.index.weekday
    df['workingday'] = np.where((df['weekday'] < 5) & (df['holiday'] == 0), 1, 0)

    return df

def main():
    
    data, var_groups, data_prep, ohe, categorical_vars = read_and_preprocess_data()
    tab1, tab2, tab3= st.tabs(['User Understanding', 'Predictive Model', 'Predict'])
    
    #Tab 1 for varaible exploration
    with tab1:
        st.title('User Demand and Variables Understanding')
        st.write('In this section we want to understand the user behavior and the relationship with the given variables.')
        cont_demand = st.container()
        cont_describe = st.container()
        cont_corr = st.container()
        cont_three_scatt = st.container()
        
        #User Demand
        with cont_demand:
            st.subheader("Total, Registered and Casual Users Evolution")
            st.write('Here is displayed the historical demand. The viewer can choose how to group the data, how to calculate the metric, the time period, and the displayed variables.')
            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7 = st.columns(3)

            demand_container = st.container()

            #Select plot variables
            plot_vars = var_groups['target'] + var_groups['ignore']
            selected_varialbes = col1.multiselect('Select variables to plot',
                                                         plot_vars,
                                                         plot_vars[0])

            #Dinamic Group by
            time_periods = {'Monthly': 'MS', 'Weekly': 'W', 'Daily': 'D', 'Hourly': 'H'}
            calculations = {'Total': 'sum', 'Average (Hourly)': 'mean', 'Maximum': 'max', 'Minimum': 'min'}

            selected_period = col2.selectbox("Select Time Period", list(time_periods.keys()))
            selected_calculation = col3.selectbox("Select Calculation", calculations)

            grouped_data = data[plot_vars].groupby(pd.Grouper(freq=time_periods[selected_period])).agg(calculations[selected_calculation])
            summ_data = data[plot_vars].groupby(pd.Grouper(freq=time_periods[selected_period])).agg('sum')
            #Date Filter
            min_date = datetime.datetime(2011,1,1)
            max_date = datetime.date(2013,1,1)

            date_filter = col4.date_input("Select Period of Time", 
                                                value = (min_date,max_date), 
                                                min_value = min_date, 
                                                max_value = max_date)

            if len(date_filter) == 2:
                if date_filter[0] == date_filter[1]:
                    grouped_data = grouped_data.loc[date_filter[0]:date_filter[1] + datetime.timedelta(days=1)]
                    summ_data = summ_data.loc[date_filter[0]:date_filter[1] + datetime.timedelta(days=1)]
                else:
                    grouped_data = grouped_data.loc[date_filter[0]:date_filter[1]]
                    summ_data = summ_data.loc[date_filter[0]:date_filter[1]]
            elif len(date_filter) == 1:
                grouped_data = grouped_data.loc[date_filter[0]:date_filter[0] + datetime.timedelta(days=1)]
                summ_data = summ_data.loc[date_filter[0]:date_filter[0] + datetime.timedelta(days=1)]
            else:
                st.write(date_filter)
            
            #Summary Metrics
            col5.metric("Total Users", f'{summ_data["cnt"].sum():,}')
            col6.metric(f"Average Users {selected_period}",f'{summ_data["cnt"].sum() / len(summ_data):,.2f}')
            col7.metric("% Registered Users", f'{summ_data["registered"].sum()/summ_data["cnt"].sum():.2%}')

            ## PLOT FIGURE 1 ##
            fig1 = px.line(
                grouped_data,
                x=grouped_data.index,
                y=selected_varialbes,
                title="{} Number of Users - {}".format(selected_calculation, selected_period),
                template="none",
            )

            fig1.update_xaxes(title="Date")
            fig1.update_yaxes(title="Number of Users")
            fig1.update_traces(
                mode="lines",
                marker_size=10,
                line_width=3,
                error_y_color="gray",
                error_y_thickness=1,
                error_y_width=10,
            )

            st.plotly_chart(fig1, use_container_width=True)

        with cont_describe:
            #Description of basic stats for each varaible
            st.subheader("Variable Description")
            st.write('Here is displayed a general description of all variables.')
            st.dataframe(data.describe(), use_container_width=True)

        with cont_corr:
            #Correlations
            st.subheader("Variable Correlations")
            st.write('Here is displayed the correlation between variables. All variables, including categorical, are displayed. On the right, the variables sorted by the biggest absolute correlation with the Total Demand.')
            col5, col6 = st.columns([8, 2])
            
            #Correlation Heatmap
            corr = data.corr()
            fig0 = px.imshow(corr, x=corr.index, y=corr.columns,
                            labels=dict(color='Correlation'), color_continuous_scale='Viridis')
            fig0.update_layout(title='Correlation of all variables')
            col5.plotly_chart(fig0, use_container_width=True)
            
            #Correlation With Target Variable
            corr_matx = data[var_groups['numerical']+var_groups['catergorical']].corrwith(data['cnt']).to_frame()
            corr_matx['abs_corr'] = corr_matx[0].abs()
            corr_matx.reset_index(inplace=True)
            corr_matx = corr_matx.sort_values('abs_corr', ascending=False)
            col6.write("**Absolute Correlation with Total Demand**")
            col6.dataframe(corr_matx, use_container_width=True)
        
        with cont_three_scatt:
            #Three user interactive Scatterplots
            st.subheader("Variable Correlations")
            st.write('Here there are 3 scatter plots where the user can pick any 3 variables to plot vs the Total User demand.')
            col7, col8, col9 = st.columns(3)
            scatt_vars = list(data.columns.values)
            
            var_1 = col7.selectbox("Select Variable 1", scatt_vars, index = 8)
            var_2 = col8.selectbox("Select Variable 2", scatt_vars, index = 3)
            var_3 = col9.selectbox("Select Variable 3", scatt_vars, index = 2)
            
            fig1 = px.scatter(x=data[var_1], y=data['cnt'], labels={'x': var_1, 'y': 'Total users'}, title='Total Users vs {}'.format(var_1))
            col7.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.scatter(x=data[var_2], y=data['cnt'], labels={'x': var_2, 'y': 'Total users'}, title='Total Users vs {}'.format(var_2))
            col8.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(x=data[var_3], y=data['cnt'], labels={'x': var_3, 'y': 'Total users'}, title='Total Users vs {}'.format(var_3))
            col9.plotly_chart(fig3, use_container_width=True)            
    
    #Tab 2 has description of models
    with tab2:
        st.title('Prediction Models')
        st.write('In this section we want to show the results of the models for hourly predictions. We focus the project con predicting the total users, so the target variable is **"cnt"**. After choosing a model and making the predictions, we split the data according to the week-day and the hour.')
        st.write('We try with three models: **Linear Regression**, **Decision Tree** and **XGBoost**.')
        
        #Split of data
        X_train = data_prep.loc[(data_prep.index < '2012-08-09')].drop('cnt', axis=1)
        y_train = data_prep.loc[(data_prep.index < '2012-08-09')]['cnt']

        X_val = data_prep.loc[(data_prep.index < '2012-10-20') & (data_prep.index >= '2012-08-09')].drop('cnt', axis=1)
        y_val = data_prep.loc[(data_prep.index < '2012-10-20') & (data_prep.index >= '2012-08-09')]['cnt']

        X_test = data_prep.loc[(data_prep.index >= '2012-10-20')].drop('cnt', axis=1)
        y_test = data_prep.loc[(data_prep.index >= '2012-10-20')]['cnt']
        y_test_registered = data.loc[(data.index >= '2012-10-20')]['registered']
        y_test_casual = data.loc[(data.index >= '2012-10-20')]['casual']

        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        
        model_lr, grid_dt, model_xg = read_models()
        
        #Predict
        test_pred_lr = model_lr.predict(X_test)
        test_pred_dt = grid_dt.predict(X_test)
        test_pred_xg = model_xg.predict(X_test)

        #Metrics
        MAPE_lr_test = metric_1(y_test, test_pred_lr)
        MSE_lr_test = metric_2(y_test, test_pred_lr)
        MAPE_dt_test = metric_1(y_test, test_pred_dt)
        MSE_dt_test = metric_2(y_test, test_pred_dt)
        MAPE_xg_test = metric_1(y_test, test_pred_xg)
        MSE_xg_test = metric_2(y_test, test_pred_xg)
        
        #Results
        final_results = pd.DataFrame.from_dict({'model': ['linear_reg','decision_tree','xgboost'],
                                        'MAPE':[MAPE_lr_test, MAPE_dt_test, MAPE_xg_test],
                                        'MSE':[MSE_lr_test, MSE_dt_test, MSE_xg_test]})
        
        #Containers for each model
        cont_day_hour = st.container()
        cont_linear_reg = st.container()
        cont_dec_tree = st.container()
        cont_xgboost = st.container()
        
        with cont_day_hour:
            col1, col2, col3 = st.columns(3)
            
            col1.subheader('General model results')
            col1.write('The following table shows the results of each model. These results were found testing the 10% of the data.')
            col1.write(final_results, use_container_width=True)
            
            col2.subheader('Date - Hour split by type of user')
            col2.write('As we are only forecasting the total users, we need to calculate before the split of total users between Casual and Registered, based on historical data.')
            hour_dist = data.groupby(['weekday', 'hr'])[['registered','casual']].sum()
            hour_dist['perc_casual'] = (hour_dist['casual'])/(hour_dist['casual']+hour_dist['registered'])
            hour_dist['perc_registered'] = (hour_dist['registered'])/(hour_dist['casual']+hour_dist['registered'])
            col2.write(hour_dist, use_container_width=True)
            
            col3.subheader('Temperature Changes over the Day')
            col3.write('Additionally, for the prediction, the model requires a mean temperature input. The user will provide this input and the model will calculate the hourly temperature based on the following distribution.')
            daily_avg_temp = data.groupby(data.index.date)['temp'].transform('mean')
            new_frame = data[['hr','temp']]
            new_frame['tmp_diff'] = new_frame['temp'] - daily_avg_temp
            summary_table = new_frame.groupby('hr')['tmp_diff'].mean().reset_index()
            col3.write(summary_table, use_container_width=True)
        
        
        with cont_linear_reg:
            #Linear Regression
            st.subheader('Linear Regression Model')
            st.write(f'The first model for the hourly prediction. MAPE: {MAPE_lr_test:.2f}, MSE: {MSE_lr_test:.2f}')
            
            df_lr = pd.DataFrame(y_test.copy())
            df_lr['pred'] = test_pred_lr
            st.line_chart(df_lr, use_container_width=True)
            
            with st.expander("Linear Regression Model Details"):
                col1, col2 = st.columns(2)
                
                feature_lr = model_lr.coef_

                importance_dt = pd.DataFrame({'Feature': X_test.columns, 'Coeficients': feature_lr[0]})
                importance_dt = importance_dt.sort_values(by='Coeficients', ascending=False)

                col1.dataframe(importance_dt, use_container_width=True)
                col2.write('**Model Parameters:**')
                col2.write(model_lr.get_params())
            
        
        with cont_dec_tree:
            #Decision Tree
            st.subheader('Decision Tree Model')
            st.write(f'The first model for the hourly prediction. MAPE: {MAPE_dt_test:.2f}, MSE: {MSE_dt_test:.2f}')
            
            df_dt = pd.DataFrame(y_test.copy())
            df_dt['pred'] = test_pred_dt
            st.line_chart(df_dt, use_container_width=True)

            with st.expander("Decision Tree Model Details"):
                col1, col2 = st.columns(2)
                
                best_model = grid_dt.best_estimator_
                feature_dc = best_model.feature_importances_

                importance_dt = pd.DataFrame({'Feature': X_test.columns, 'Importance': feature_dc})
                importance_dt = importance_dt.sort_values(by='Importance', ascending=False)

                fig = px.bar(importance_dt.head(10), x='Importance', y='Feature', orientation='h', labels={'Importance': 'Importance Score', 'Feature': 'Feature'})
                fig.update_layout(title='Decision Tree Variable Importance', xaxis_title='Importance Score', yaxis_title='Feature')
                col1.plotly_chart(fig, use_container_width=True)
                col2.write('**Model Parameters:**')
                col2.write(best_model.get_params())
        
        with cont_xgboost:
            #Xgboost
            st.subheader('XGBoost Model')
            st.write(f'The first model for the hourly prediction. MAPE: {MAPE_xg_test:.2f}, MSE: {MSE_xg_test:.2f}')
            
            df_xg = pd.DataFrame(y_test.copy())
            df_xg['pred'] = test_pred_lr
            st.line_chart(df_xg, use_container_width=True)
            
            with st.expander("XGBoost Model Details"):
                col1, col2 = st.columns(2)

                importance_xg = pd.DataFrame(list(model_xg.get_booster().get_score().items()), columns=['Feature', 'Importance'])
                importance_xg = importance_xg.sort_values(by='Importance', ascending=False)
            
                fig = px.bar(importance_xg.head(10), x='Importance', y='Feature', orientation='h', labels={'Importance': 'Importance Score', 'Feature': 'Feature'})
                fig.update_layout(title='XGBoost Variable Importance', xaxis_title='Importance Score', yaxis_title='Feature')
                col1.plotly_chart(fig, use_container_width=True)
                col2.write('**Model Parameters:**')
                col2.write(model_xg.get_xgb_params())
    #Tab 3 is for prediction
    with tab3:
        st.title('Predict')
        st.write('Use the parameters to predict user demand for an specific date')

        col1, col2, col3, col7 = st.columns(4)
        col4, col5, col6= st.columns(3)
        
        #Parameter Input. 7 in total.
        selected_model = col1.selectbox("Select Model to Predict", ['Decision Tree', 'XGBoost'])
        holiday = col3.toggle('Holiday?')
        prediction_date = col2.date_input('Select Date for Predictions', min_value = datetime.datetime(2013,1,1), max_value = datetime.datetime(2024,12,31))
        general_weather = col7.selectbox("Select General Climate Forecast", ['Clear', 'Misty, Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'])
        mean_temperature_forecast = col4.slider("Select Mean Temperature C° Forecast", -10, 42, 15)
        mean_humidity_forecast = col5.slider("Select Mean Humidity Forecast", 0, 100, 60)
        mean_windspeed_forecast = col6.slider("Select Mean Windspeed C° Forecast", 0, 67, 0)
        
        summary_table['final_temp'] = 0.019607843137254*mean_temperature_forecast+0.196078431372549 + summary_table['tmp_diff']
        
        #Forecast Framework construction
        x_forecast = dataframe_prediction(prediction_date)
        x_forecast['holiday'] = 1 if holiday else 0
        x_forecast['weathersit'] = 1 if general_weather == 'Clear' else ( 2 if general_weather == 'Misty, Cloudy' else (3 if general_weather == 'Light Rain/Snow' else 4))
        x_forecast['temp'] = summary_table['final_temp'].values
        x_forecast['atemp'] = x_forecast['temp']
        x_forecast['hum'] = mean_humidity_forecast/100
        x_forecast['windspeed'] = 0.014925373134328358*mean_windspeed_forecast
        x_forecast['climate_perception'] = 1 if general_weather == 'Clear' else ( 2 if general_weather == 'Misty, Cloudy' else (3 if general_weather == 'Light Rain/Snow' else 3))
        x_forecast['temperature_perception'] = x_forecast['temp']

        
        #Same OHE for forecast
        x_forecast.reset_index(inplace=True)
        x_pred = x_forecast.copy()

        dat_ohe = pd.DataFrame(ohe.transform(x_pred[categorical_vars]))
        dat_ohe.columns = ohe.get_feature_names_out()
        x_pred = x_pred.drop(categorical_vars, axis=1)
        x_pred = pd.concat((x_pred, dat_ohe), axis=1)

        #Set the date back as index in both datasets
        x_pred.set_index('index', inplace=True)
        
        #Select the model
        if selected_model == 'Linear Regression':
            array_prediction = model_lr.predict(x_pred)
        elif selected_model == 'Decision Tree':
            array_prediction = grid_dt.predict(x_pred)
        else:
            array_prediction = model_xg.predict(x_pred)
        
        #Graph the results
        dt_pred = pd.DataFrame(index=x_pred.index)
        dt_pred['total_prediction'] = array_prediction
            #Dont know how this is working. Dont touch
        dt_pred['registered_prediction'] = array_prediction * hour_dist['perc_registered'].head(len(dt_pred)).values
            #Same
        dt_pred['casual_prediction'] = array_prediction * hour_dist['perc_casual'].head(len(dt_pred)).values
        st.line_chart(dt_pred)
        
        #Download Data
        st.download_button('Download Prediction in CSV', 
                           dt_pred.to_csv().encode('utf-8'), 
                           file_name=f'{prediction_date}_user_prediction.csv', 
                           mime='text/csv')
    
        
if __name__ == "__main__":
    # This is to configure some aspects of the app
    st.set_page_config(
        layout="wide", page_title="Bike Sharing DC", page_icon=":bike:"
    )

    # Write titles in the main frame and the side bar
    st.markdown("<h1 style='text-align: center; color: black;'>Bike Sharing in Washington DC </h1>", unsafe_allow_html=True)
    #st.sidebar.title("Options")

    # Call main function
    main()
