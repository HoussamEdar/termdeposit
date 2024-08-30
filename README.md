Project: Marketing Analysis


The main issue we're addressing is that numerous banks are interested in identifying the customer 
segments most likely to take up an offer to better target their efforts effectively
and efficiently invest their resources where it matters most.
To start off our analysis process.
Gain insights into the problem at hand more clearly and thoroughly 
understand the data set provided by running various classifiers and exploring different strategies to enhance their performance, 
for predicting campaign responses. The original dataset consists of the banking dataset comprising 41188 rows and 21 characteristics/features
in total most of which were categorical in nature prompting us to convert the data and introduce variables to delve deeper into each categorical variable present, 
within it. The specific categorical variables included 'job' 'marital' 'education' 'default' 'housing' 'loan' 'contact' 'month' 'day_of_week' and 'poutcome'.

Our main research question is:

To predict whether a customer avails term deposit or not.

This  data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution.
The classification goal is to predict if the client will subscribe a term deposit (variable y)


Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")
  <img width="437" alt="image" src="https://github.com/user-attachments/assets/ecd69a9e-3818-437c-a6ab-7d21aa926c91">
  <img width="512" alt="image" src="https://github.com/user-attachments/assets/6a67c462-902d-4273-897a-3d1b1e53fe2c">


