import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from word2number import w2n
import joblib
import streamlit as st

st.set_page_config(page_title="Multiple Linear Regression", page_icon="📈")

st.title("Multiple Linear Regression 📈")
st.subheader("Exercise:")
st.write("""
In exercise folder (same level as this notebook on github) there is hiring.csv. This file contains hiring statics for a firm such as experience of candidate, his written test score and personal interview score. Based on these 3 factors, HR will decide the salary. Given this data, you need to build a machine learning model for HR department that can help them decide salaries for future candidates. Using this predict salaries for following candidates:

1) 2 yr experience, 9 test score, 6 interview score

2) 12 yr experience, 10 test score, 10 interview score
""")


st.subheader("Let's try to solve this:")
st.write("Here we have given data set")
data = pd.read_csv('D:\Machine_learning\multi_reg\hiring.csv')
st.table(data)

st.subheader("""1) First we are going to handel missing values""")
st.write("Fill missing values of 'experience' with zero and 'test_score(out of 10)' with the mean of column")

st.code("""
data['experience']=data['experience'].fillna('zero')
data['test_score(out of 10)']=data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean())        
""")

data['experience']=data['experience'].fillna('zero')
data['test_score(out of 10)']=data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean())  

st.table(data)

st.subheader("""2) Now we have to convert alphabatic value of 'experience' into numeric form""")
st.code("""
        number = data['experience'].to_list()
print(number)
lis = []

for num in number:
    lis.append(w2n.word_to_num(num))

data['experience']=pd.Series(lis)
data
        """)

number = data['experience'].to_list()
print(number)
lis = []

for num in number:
    lis.append(w2n.word_to_num(num))

data['experience']=pd.Series(lis)


st.write("Now we get cleaned dataset to train our model")
st.table(data)

st.subheader("""3) Now we we have to train the model (Multiple linear Regression)""")
st.write("first let's visualize the relationship between variables")

fig, ax = plt.subplots()
ax.scatter(data['experience'], data['salary($)'], color='b', label="experience")  # Experience vs. Salary
ax.scatter(data["test_score(out of 10)"], data['salary($)'], color='g', label="test_score(out of 10)")  # Test Score vs. Salary
ax.scatter(data["interview_score(out of 10)"], data['salary($)'], color='r', label="interview_score(out of 10)")  # Interview Score vs. Salary
ax.set_xlabel('Features')
ax.set_ylabel('Salary')
st.pyplot(fig)

st.write("🔵experience  🟢test_score(out of 10)  🔴interview_score(out of 10)")

st.write('Training linear regression model')
st.code("""
        from sklearn import linear_model
        reg = linear_model.LinearRegression()
        model = reg.fit(data[['experience','test_score(out of 10)','interview_score(out of 10)']],data[['salary($)']])
        """)

model = joblib.load("D:\Machine_learning\multi_reg\model.joblib")

st.subheader("Solution :")
st.write("1) 2 yr experience, 9 test score, 6 interview score")
st.write(model.predict([[2,9,6]]))

st.write("2) 12 yr experience, 10 test score, 10 interview score")
st.write(model.predict([[12,10,10]]))

st.subheader("Predictions:")

x = st.number_input(label="experience (yr)")
if x<0:
    st.warning("Experience should be a whole number")

y = st.number_input(label="test_score(out of 10)")
if y>10 or y<0:
    st.warning("Score should be between 0 to 10")

z = st.number_input(label="interview_score(out of 10)")
if z>10 or z<0:
    st.warning("Score should be between 0 to 10")

btn = st.button(label="Predict")

if btn:
    st.write(model.predict([[x,y,z]]))