from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลดอกไม้")
#st.image("./img/kairung022.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("Versicolor")
   st.image("./img/katawut.jpeg")

with col2:
   st.header("Verginiga")
   st.image("./img/heart-attack.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")


age = st.slider("กรุณาเลือกข้อมูล petal.length")
sex = st.slider("กรุณาเลือกข้อมูล petal.width")
cp_type = st.number_input("กรุณาเลือกข้อมูล sepal.length")
r_bp = st.number_input("กรุณาเลือกข้อมูล sepal.width")
Ch = st.number_input("กรุณาเลือกข้อมูล sepal.width")
f_bs = st.number_input("กรุณาเลือกข้อมูล sepal.width")
r_ecg = st.number_input("กรุณาเลือกข้อมูล sepal.width")
m_hr = st.number_input("กรุณาเลือกข้อมูล sepal.width")
ea = st.number_input("กรุณาเลือกข้อมูล sepal.width")
Old =  st.number_input("กรุณาเลือกข้อมูล sepal.width")
st_s = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   X = dt.drop('variety', axis=1)
   y = dt.variety   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[age, sex, cp_type, r_bp, Ch, f_bs, r_ecg, m_hr, ea, Old, st_s]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./img/heart-attack.jpg")
   else:
    st.image("./img/katawut.jpeg")
else:
    st.write("ไม่ทำนาย")
