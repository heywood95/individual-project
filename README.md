# Individual-Project
 
# Project Description
 
The worldwide governance indicators will be investigated using a created target variable to predict what factors impact the rankings of overall governance for over 200 countries and territories over the period 1996–2021, using regression techniques, and make predictions on overall governance. I have decided to look into six different areas that may affect overall governance.
 
# Project Goal
 
* Construct an ML Regression model that predicts worldwide governance using features derived from the The Worldwide Governance Indicators (WGI) project.
* Find the key drivers of worldwide governance. 
* Deliver a report that explains what steps were taken, why and what the outcome was.
* Make recommendations on what works or doesn't work in predicting overall worldwide governance.
 
# Initial Thoughts
 
My initial hypothesis is that overall governance will be impacted differently depending on which of the six indicators are present.
 
# The Plan
 
* Aquire data from databank.worldbank.org via the Worldwide Governance Indicators (WGI) project
 
* Prepare data
   * Cleaned up data
       * Droped the null values.
       * Replaced zero placeholders with zero.
       * Turned the numerical columns that were objects into floats . 
       * Dropped unnecessary and redundant columns.
       * Renamed confusing and exceedingly long column names . 
       * Created a new column 'overall_govt' that encompases governmental rankings into one overall ranking that could be used for the target variable.
       * split the data.  
 
* Explore data in search of drivers of overall governance
   * Answer the following initial questions
       * Does the rule of law strengthen overall government?
       * Does controlling corruption effect the overall govt.?
       * How strong of a correlation is regulatory quality and political stability in relation to overall govt.?  
       * What is the relationship between voice and accountability and effective government in conjunction with overall government?  
      
* Develop a Model to predict the rankings of overall governance
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest validate and difference accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|'date'|	 Year of the specified country's ranking|
|'control_corruption'|	Control of Corruption Captures perceptions of the extent to which public power is exercised for private gain, including both petty and grand forms of corruption, as well as "capture" of the state by elites and private interests.|
|'govt_effective'| Government Effectiveness captures perceptions of the quality of public services, the quality of the civil service and the degree of its independence from political pressures, the quality of policy formulation and implementation, and the credibility of the government's commitment to such policies. Percentile rank indicates the country's rank among all countries covered by the aggregate indicator, with 0 corresponding to lowest rank, and 100 to highest rank.|
|'political_stability'|	Political Stability and Absence of Violence/Terrorism measures perceptions of the likelihood of political instability and/or politically-motivated violence, including terrorism.|
|'regulatory_quality'|	Regulatory Quality captures perceptions of the ability of the government to formulate and implement sound policies and regulations that permit and promote private sector development.| 
|'rule_of_law'|	Rule of Law captures perceptions of the extent to which agents have confidence in and abide by the rules of society, and in particular the quality of contract enforcement, property rights, the police, and the courts, as well as the likelihood of crime and violence.|
|'voice_accountability'|	Voice and Accountability captures perceptions of the extent to which a country's citizens are able to participate in selecting their government, as well as freedom of expression, freedom of association, and a free media.| 
|'overall_govt'|	Final ranking taking all six of the governmental features and producing a single score from 0-100.|
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from https://databank.worldbank.org/source/worldwide-governance-indicators
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions:
* The overall governance of countries depends on six factors evaluated from various sources, all factors connect strongly with overall governance, but careful consideration should be taken when exporing the impact of these factors and how the countries are ranked.
* Rule of Law and Government effictiveness appear to be more important in the rankings of governments than other humanitarian considerations like political stability, which incoporates acts of terrorism and violence, and voice and accountability, which include media outlets and other oragnizations that hold governments accountable through freedom os speech.
* Although countries recieve high rankings for overall governance, they can also recieve low rankings for political stability and voice and accountability. However, the governments, for the most part, do not recieve high rankings if their rule of law, control of corruption, or government effectiveness ratings are lower.
 
# Recommendations
* Further analysis on the importance of governmental/legislative impacts on the rankings versus humanitarian effects on how countries are ranked should be examined to avoid ranking governments higher that do not incorporate political stability or accountability.
* This findings indicate that countries are perceived as having a high level of governance if they maintain "law and order," but an emphasis on humanitarian initiatives and absence of violence should be given equal weight, if not more importance.
* These six factors should be broken down further into sub-categories to gain a better understanding of the rankings and other factors included, such as environmental, to gain a more accurate rating of each of the governments.
