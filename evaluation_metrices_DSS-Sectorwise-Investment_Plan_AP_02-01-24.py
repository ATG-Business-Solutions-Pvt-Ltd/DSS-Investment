# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Set the title of your app
st.title("Your Investment Companion")

# Add an option to upload data
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Get the unique companies
    companies = df['Company '].unique()

    # Ask the user to choose a company
    company = st.selectbox('Please choose a company:', companies)

    # Get the unique categories for the chosen company
    company_categories = df[df['Company '] == company]['Category'].unique()

    # Ask the user to choose a category
    category = st.selectbox('Please choose a category:', company_categories)

    # Filter data for the chosen company and category
    company_data = df[(df['Category'] == category) & (df['Company '] == company)]

    # Plotting
    st.write("### **This Section will guide you with three important financial markers:**") 
    def plot_company_category(company_data):
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(company_data['Year'], company_data['Price/Earning Ratio'], label='Price/Earning Ratio')
      ax.plot(company_data['Year'], company_data['Earning Per Share'], label='Earning Per Share')
      ax.plot(company_data['Year'], company_data['Gross Margin'], label='Gross Margin')
      ax.set_xlabel('Year')
      ax.set_ylabel('Value')
      ax.set_title(f'Change in Financial Metrics Over Time for {company} in {category} Category')
      
      ax.legend(fontsize='small')
      return fig
    
    # Now you can call this function with the company_data DataFrame
    fig=plot_company_category(company_data)
    st.pyplot(fig)

    def suggest_investment(company_data):
      # Calculate the trend of the last two years for each ratio
      pe_trend = company_data['Price/Earning Ratio'].iloc[-1] - company_data['Price/Earning Ratio'].iloc[-2]
      eps_trend = company_data['Earning Per Share'].iloc[-1] - company_data['Earning Per Share'].iloc[-2]
      gm_trend = company_data['Gross Margin'].iloc[-1] - company_data['Gross Margin'].iloc[-2]

      # Basic investment suggestion based on the trend
      if pe_trend > 0 and eps_trend > 0 and gm_trend > 0:
          return "The Price/Earning Ratio, Earnings Per Share, and Gross Margin are all trending upwards for this company and category. This could be a positive sign for potential investment."
      elif pe_trend < 0 and eps_trend < 0 and gm_trend < 0:
          return "The Price/Earning Ratio, Earnings Per Share, and Gross Margin are all trending downwards for this company and category. You may want to exercise caution when considering this as an investment."
      else:
          return "The financial indicators show mixed trends. It's recommended to conduct further analysis or consult with a financial advisor before making an investment decision."
    
    #To calculate error matrics
    def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Now you can call this function with the company_data DataFrame
    investment_suggestion = suggest_investment(company_data)
    st.write("#### **The suggestions are based on analysing last two years trend of the financial markers**")
    st.write(investment_suggestion)

    st.write("### **Investment Suggestions based on Forecasting of Market Cap and Gross Profit for the next year:**")
    # Check if data exists for the chosen company and category
    if not company_data.empty:
        # Define the parameters
        parameters = ['Gross Margin', '(CFO-CFI)/EBIDTA', 'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'Free Cash Flow per Share', 'Price/Earnings Ratio', 'Earnings Per Share', 'Gross Margin', '(CFO-CFI)/EBIDTA']
        # Print the parameters
        st.write("\nThe following financial parameters are being used for the forecasting:")
        for i, parameter in enumerate(parameters, 1):
            st.write(f"{i}. {parameter}")
        # Fit the ARIMA model for each parameter and forecast
        for parameter in parameters:
            # Split the data into train and test sets
            train_data = company_data[company_data['Year'] < 2022]
            test_data = company_data[company_data['Year'] == 2022]

            # Fit the ARIMA model on the training data
            model = ARIMA(train_data['Gross Profit'], order=(1,1,1))
            model_fit_profit = model.fit()
            # Predict the parameter for the next year
            forecast_gross_profit = model_fit_profit.forecast(steps=12)  # Forecast the next 12 months
            # Get the 12th month forecasted value
            forecast_value_profit_12th_month = forecast_gross_profit.iloc[-1]
            # Calculate the RMSE
            mse = mean_squared_error([test_data['Gross Profit'].values[0]], [forecast_gross_profit.iloc[-1]])
            rmse_profit = sqrt(mse)
            # Calculate the MAPE
            mape_profit = mean_absolute_percentage_error([test_data['Gross Profit'].values[0]],[forecast_gross_profit.iloc[-1]])


            # Fit the ARIMA model for Market Cap
            model_cap = ARIMA(train_data['Market Cap(in B USD)'], order=(1,1,1))
            model_fit_cap = model_cap.fit()
            # Predict the Market Cap for the next year
            forecast_cap = model_fit_cap.forecast(steps=12) # Forecast the next 12 months
            # Get the 12th month forecasted value
            forecast_value_cap_12th_month = forecast_cap.iloc[-1]
            # Calculate the RMSE
            mse = mean_squared_error([test_data['Market Cap(in B USD)'].values[0]], [forecast_cap.iloc[-1]])
            rmse_cap = sqrt(mse)
            # Calculate the MAPE
            mape_cap = mean_absolute_percentage_error([test_data['Market Cap(in B USD)'].values[0]], [forecast_cap.iloc[-1]])


        # Print the company, category, predicted Gross Profit, and predicted Market Cap
        st.write("#### **Inferences based on the analysis**")
        st.write(f"##### **Company: {company}, Category: {category}**")
        st.write(f"**Based on the financial markers the predicted Gross Profit for next year: {forecast_value_profit_12th_month}**")
        st.write(f"**Based on the financial markers the predicted Market Cap for next year: {forecast_value_cap_12th_month}**")
        
        #st.write(f'RMSE for Gross Profit: {rmse_profit:.3f}')
        #st.write(f'MAPE for Gross Profit: {mape_profit:.2f}%')

        #st.write(f'RMSE for Market Cap: {rmse_cap:.3f}')
        #st.write(f'MAPE for Market Cap: {mape_cap:.2f}%')

        # Compare with other categories of the same company
        other_categories = df[(df['Company '] == company) & (df['Category'] != category)]['Category'].unique()
        for other_category in other_categories:
            other_data = df[(df['Category'] == other_category) & (df['Company '] == company)]
        #print(other_data)
            train_data = other_data[other_data['Year'] < 2022]
            test_data = other_data[other_data['Year'] == 2022]

            other_model_profit = ARIMA(train_data['Gross Profit'], order=(1,1,1))
            other_model_fit_profit = other_model_profit.fit()
            other_forecast_profit = other_model_fit_profit.forecast(steps=12)
            # Get the 12th month forecasted value
            forecast_otherdata_profit_12th_month = other_forecast_profit.iloc[-1]
            # Calculate the MAPE
            mape_profit_otherdata = mean_absolute_percentage_error([test_data['Market Cap(in B USD)'].values[0]], other_forecast_profit.iloc[-1])


            other_model_cap = ARIMA(train_data['Market Cap(in B USD)'], order=(1,1,1))
            other_model_fit_cap = other_model_cap.fit()
            other_forecast_cap = other_model_fit_cap.forecast(steps=12)
            # Get the 12th month forecasted value
            forecast_otherdata_cap_12th_month = other_forecast_cap.iloc[-1]
            # Calculate the MAPE
            mape_cap_otherdata = mean_absolute_percentage_error([test_data['Market Cap(in B USD)'].values[0]], other_forecast_cap.iloc[-1])

            if forecast_otherdata_profit_12th_month > forecast_value_profit_12th_month and forecast_otherdata_cap_12th_month > forecast_value_cap_12th_month:
                print(f"The {other_category} category of {company} is expected to have a higher Gross Profit and Market Cap next year. You may want to consider investing in that category.")
                
    else:
         st.write("No data available for the chosen company and category.")


    #st.write(f'MAPE for Other_Gross Profit: {mape_profit_otherdata:.2f}%')
    #st.write(f'MAPE for Other_Market Cap: {mape_cap_otherdata:.2f}%')


    #Same sector choice
    # Get all unique companies
    all_companies = df['Company '].unique()

    # Initialize a dictionary to store the forecasted values for each company
    company_forecasts = {}

    # Loop through all companies
    
    for comp in all_companies:
        # Filter data for the chosen category and company
        comp_data = df[(df['Category'] == category) & (df['Company '] == comp)]

        # Check if data exists for the chosen company and category
        if not comp_data.empty:
          train_data = comp_data[comp_data['Year'] < 2022]
          test_data = comp_data[comp_data['Year'] == 2022]

          # Fit the ARIMA model for Gross Profit and forecast
          comp_model_profit = ARIMA(train_data['Gross Profit'], order=(1,1,1))
          comp_model_fit_profit = comp_model_profit.fit()
          comp_forecast_profit = comp_model_fit_profit.forecast(steps=12)
          # Get the 12th month forecasted value
          forecast_compdata_profit_12th_month = comp_forecast_profit.iloc[-1]
          # Calculate the MAPE
          mape_profit_compdata = mean_absolute_percentage_error([test_data['Gross Profit'].values[0]], comp_forecast_profit.iloc[-1])  

          # Fit the ARIMA model for Market Cap and forecast
          comp_model_cap = ARIMA(train_data['Market Cap(in B USD)'], order=(1,1,1))
          comp_model_fit_cap = comp_model_cap.fit()
          comp_forecast_cap = comp_model_fit_cap.forecast(steps=12)
          # Get the 12th month forecasted value
          forecast_compdata_cap_12th_month = comp_forecast_cap.iloc[-1]
          # Calculate the MAPE
          mape_cap_compdata = mean_absolute_percentage_error([test_data['Market Cap(in B USD)'].values[0]], comp_forecast_cap.iloc[-1])


          # Store the forecasted values in the dictionary
          company_forecasts[comp] = (forecast_compdata_profit_12th_month, forecast_compdata_cap_12th_month)

    #st.write(f'MAPE for Other_Gross Profit: {mape_profit_compdata:.2f}%')
    #st.write(f'MAPE for Other_Market Cap: {mape_cap_compdata:.2f}%')        

    # Find the company with the highest forecasted Gross Profit and Market Cap
    best_company = max(company_forecasts, key=company_forecasts.get)

    # Print the best company to invest in
    st.write(f"##### **The best company to invest in the {category} category based on the predicted Gross Profit and Market Cap is: {best_company}**")

