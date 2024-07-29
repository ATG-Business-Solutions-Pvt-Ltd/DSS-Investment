 #pip install transformers sentencepiece nltk matplotlib scikit-learn streamlit torch xlrd openpyxl statsmodels langchain langchain-community accelerate

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import transformers
import torch
nltk.download('punkt')
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
#from transformers import LongT5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate


# Load the data
df = pd.read_excel("Financial Statement-modified.xls")
df1 = pd.read_excel("financial_news.xlsx", sheet_name='News')

# Add the 'financial_news' column from df1 to df
df['financial_news'] = df1['financial_news']

# Get the unique companies
companies = df['Company'].unique()

# Ask the user to choose a company
company = st.selectbox('Please choose a company:', companies)

# Get the unique categories for the chosen company
company_categories = df[df['Company'] == company]['Category'].unique()

# Ask the user to choose a category
category = st.selectbox('Please choose a category:', company_categories)

# Filter data for the chosen company and category
company_data = df[(df['Category'] == category) & (df['Company'] == company)]


# Set the title of your app
st.title("Your Investment Companion")


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

# Generate the report
report = []
comparison_with_other_categories=""
if not company_data.empty:
        # Define the parameters
        parameters = ['Gross Margin', '(CFO-CFI)/EBIDTA', 'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'Free Cash Flow per Share', 'Price/Earnings Ratio', 'Earnings Per Share']
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
            
            # Fit the ARIMA model for Market Cap
            model_cap = ARIMA(train_data['Market Cap(in B USD)'], order=(1,1,1))
            model_fit_cap = model_cap.fit()
            # Predict the Market Cap for the next year
            forecast_cap = model_fit_cap.forecast(steps=12) # Forecast the next 12 months
            # Get the 12th month forecasted value
            forecast_value_cap_12th_month = forecast_cap.iloc[-1]


        # Compare with other categories of the same company
        other_categories = df[(df['Company'] == company) & (df['Category'] != category)]['Category'].unique()
        for other_category in other_categories:
            other_data = df[(df['Category'] == other_category) & (df['Company'] == company)]
            other_model_profit = ARIMA(other_data['Gross Profit'], order=(1,1,1))
            other_model_fit_profit = other_model_profit.fit()
            other_forecast_profit = other_model_fit_profit.forecast(steps=1)
            other_forecast_value_profit = float(other_forecast_profit)
            other_model_cap = ARIMA(other_data['Market Cap(in B USD)'], order=(1,1,1))
            other_model_fit_cap = other_model_cap.fit()
            other_forecast_cap = other_model_fit_cap.forecast(steps=1)
            other_forecast_value_cap = float(other_forecast_cap)
            if other_forecast_value_profit > forecast_value_profit_12th_month and other_forecast_value_cap > forecast_value_cap_12th_month:
                #st.write(f"The {other_category} category of {company} is expected to have a higher Gross Profit and Market Cap next year. You may want to consider investing in that category.")
 
                comparison_with_other_categories += f"The {other_category} category of {company} is expected to have a higher Gross Profit and Market Cap next year. You may want to consider investing in that category.\n"
        report.append(comparison_with_other_categories)
        

else:
         st.write("No data available for the chosen company and category.")


#Same sector choice
# Get all unique companies
all_companies = df['Company'].unique()

# Initialize a dictionary to store the forecasted values for each company
company_forecasts = {}

# Loop through all companies
for comp in all_companies:
    # Filter data for the chosen category and company
    comp_data = df[(df['Category'] == category) & (df['Company'] == comp)]

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
        

        # Fit the ARIMA model for Market Cap and forecast
        comp_model_cap = ARIMA(train_data['Market Cap(in B USD)'], order=(1,1,1))
        comp_model_fit_cap = comp_model_cap.fit()
        comp_forecast_cap = comp_model_fit_cap.forecast(steps=12)
        # Get the 12th month forecasted value
        forecast_compdata_cap_12th_month = comp_forecast_cap.iloc[-1]
        

        # Store the forecasted values in the dictionary
        company_forecasts[comp] = (forecast_compdata_profit_12th_month, forecast_compdata_cap_12th_month)

    
# Find the company with the highest forecasted Gross Profit and Market Cap
best_company = max(company_forecasts, key=company_forecasts.get)

# Add the best company to the list
report.append(f"The best company to invest in the {category} category based on the predicted Gross Profit and Market Cap is: {best_company}")

    
# Data Preprocessing
company_data['financial_news'] = company_data['financial_news'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))


# Information Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(company_data['financial_news'])
feature_names = vectorizer.get_feature_names_out()
dense = X.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names)

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
company_data['sentiments'] = company_data['financial_news'].apply(lambda x: sid.polarity_scores(x))


# Add the forecasted values to the report
report.append(f"Forecasted Gross Profit for {company} in {category} for 2023: {round(forecast_value_profit_12th_month, 2)} USD")
#report.append(f"Forecasted Market Cap for {company} in {category} for 2023: {round(forecast_value_cap_12th_month, 2)} USD")

# Add the comparison with other categories to the report
#report.append(comparison_with_other_categories)

# Perform sentiment analysis on the financial news
sentiments_df = company_data['sentiments'].apply(pd.Series)
mean_sentiments = sentiments_df.mean()

# Determine the overall sentiment based on the analysis
if mean_sentiments['pos'] > mean_sentiments['neg']:
    overall_sentiment = 'positive'
elif mean_sentiments['pos'] < mean_sentiments['neg']:
    overall_sentiment = 'negative'
else:
    overall_sentiment = 'neutral'

# Add the sentiment analysis summary to the report
report.append(f"The sentiment analysis of the financial news is: {overall_sentiment}\n")
# Convert the report to a string
report_str = "\n".join(report)

st.title("\n **Brief Analysis and Comparison Outcome**")

st.write(report_str)


st.title("\n **Detailed Financial Advice**")

report1=[]
financial_news = company_data['financial_news'].iloc[0]
report1.append(f"financial news for {company} in {category}:")
report1.append(financial_news)
# You can convert it into a string using join()
report1_str = "\n".join(report1)

text = report_str + "\n" + report1_str

# Fixed LLM template prompt

fixed_prompt="""
Write a detailed financial report with the following sections. Each section should start on a new line and have a bold heading. Provide relevant information for each point:

1. **Sentiment Analysis**:
   - Briefly discuss the sentiment analysis results from the text file.

2. **Forecast Gross Profit**:
   - State the forecasted gross profit value for the next financial year.
   
3. **Comparison Report**:
   - Indicate the best category for investment in the chosen/selected company.
   - Include a comparison with other companies in the same category.

4. **Risk Analysis**:
   - Include a measure of the company's volatility (beta).
   - Provide a breakdown of the types of risk the company is exposed to (e.g., market risk, credit risk, operational risk).
   - No need to compare with competitors.

5. **Competitor Analysis**:
   - Compare the chosen company's financials and performance metrics against its main competitors.

6. **Market Trends**:
   - Analyze broader market trends impacting the company or sector.
   - Consider economic indicators, regulatory changes, or technological advancements.
   - No need to compare with competitors.

7. **Dividend Information**:
   - Provide details of the company's dividend history and policies.
   - Include a forecast of future dividend payments.
   - No need to compare with competitors.

8. **Management Analysis**:
   - Share information about the company's management team.
   - Provide background details.
   - Evaluate their performance.
   - No need to compare with competitors.

9. **Sustainability and ESG Factors**:
   - Analyze the company's environmental, social, and governance (ESG) performance.
   - Highlight factors relevant to investors.
   - No need to compare with competitors.

10. **Historical Performance**:
   - Dedicate a section to the historical performance of the company's stock.
   - Include major rises and falls.
   - No need to compare with competitors.

11. **Future Projections**:
    - Include more detailed future projections (e.g., projected revenue, net income, cash flow).
    - No need to compare with competitors.

12. **Investment Recommendation**:
    - Based on all the above factors, provide a clear investment recommendation (buy, hold, sell).

13. **Industry Analysis**:
    - Provide an overview of the industry in which the company operates.
    - Discuss industry trends, growth prospects, and challenges.
    - Consider macroeconomic factors affecting the sector.

14. **SWOT Analysis**:
    - Evaluate the company's strengths, weaknesses, opportunities, and threats.
    - Highlight key internal and external factors impacting the business.

15. **Financial Ratios**:
    - Calculate and analyze relevant financial ratios (e.g., liquidity ratios, profitability ratios, solvency ratios).
    - Compare these ratios with industry benchmarks or historical data.

16. **Cash Flow Analysis**:
    - Assess the company's cash flow statement.
    - Discuss operating, investing, and financing activities.
    - Identify any cash flow constraints or opportunities.

17. **Valuation Methods**:
    - Explore different valuation techniques (e.g., discounted cash flow, price-to-earnings ratio, price-to-book ratio).
    - Provide an estimated valuation for the company.

18. **Regulatory and Legal Factors**:
    - Discuss any legal or regulatory issues affecting the company.
    - Consider compliance, litigation, or pending regulatory changes.

19. **Geographic Expansion**:
    - Evaluate the company's geographic reach and expansion plans.
    - Discuss risks and opportunities related to international markets.

20. **Technology and Innovation**:
    - Assess the company's technological advancements and innovation strategy.
    - Consider how technology impacts its competitive edge.

21. **Corporate Social Responsibility (CSR)**:
    - Analyze the company's CSR initiatives.
    - Evaluate its impact on brand reputation and stakeholder relations.

22. **Investment Risks and Mitigation Strategies**:
    - Identify specific risks (e.g., currency risk, geopolitical risk, supply chain disruptions).
    - Propose strategies to mitigate these risks.
    
23. **Graphs and Visuals**:
    - Include relevant graphs, charts, or visual representations to support the analysis.
    - Use visuals to illustrate trends, comparisons, or key findings.

24. **Overview of the Report **:
  - Summarize the key points discussed in the report.
  - Provide a concise recommendation for potential investors.


```{text}```
 FINANCIAL REPORT:

"""

# Create a multiline text input for the template
#template_input = st.text_area("Enter your LLM template (use `{text}` as a placeholder for the combined features and comments):", height=200)

# Create a button to generate the output
if st.button("Generate Output"):
    # Replace the `{text}` placeholder in the template with the actual combined text
    template_input = fixed_prompt

    # LLM model setup
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=10000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    # Generate the LLM output
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    prompt = PromptTemplate(template=fixed_prompt, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_output=llm_chain.run(text)
    #llm_output = pipeline(template_input)[0]['generated_text']
    
    # Find the index of "FINANCIAL REPORT:"
    index = llm_output.find("FINANCIAL REPORT:")

    # Extract the part of the string after "FINANCIAL REPORT:"
    output_after = llm_output[index + len("FINANCIAL REPORT:"):]

    # Display the generated output
    st.write("**Detailed Financial Report:**")
    st.write(output_after)
