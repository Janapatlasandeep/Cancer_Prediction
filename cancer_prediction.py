import streamlit as st
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle

model = pickle.load(open("estimator_pl.pkl","rb"))



st.title("Cancer Prediction Project")

Age = st.number_input("Enter the age:",min_value = 10,max_value = 100)
Gender = st.selectbox("Select the gender:", ["Male", "Female"])
Tumor_Size= st.number_input("Enter the tumor size:",min_value = 1,max_value = 10)
Tumor_Grade= st.selectbox("Select the tumor grade:",["Nan","Low","Medium","High"])
Symptoms_Severity=st.selectbox("Select the symptoms severity:",["Nan","Mild","Moderate","Severe"])
Family_History = st.selectbox("Select if there is any family history:",["Nan","Yes","No"])
Smoking_History= st.selectbox("Select the smoking status:",["Nan","Non-Smoker","Current Smoker","Former Smoker"])
Alcohol_Consumption=st.selectbox("Select the alcohol consumption rate:",["Nan","Low","Moderate","High"])
Exercise_Frequency = st.selectbox("Select Exercise routine:",["Nan","Rarely","Regularly","Occasionally","Never"])

 
if st.button("Submit"):
    prediction = model.predict([[Age,Gender,Tumor_Size,Tumor_Grade,Symptoms_Severity,Family_History,Smoking_History,Alcohol_Consumption,Exercise_Frequency]])[0]
    st.write("The predicted value is:", prediction)
