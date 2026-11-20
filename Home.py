from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ")

col1, col2 = st.columns(2)

with col1:
    st.header("กลุ่มเสี่ยง / เป็นโรคหัวใจ")
    st.image("./img/heart-attack.jpg")

with col2:
    st.header("กลุ่มปกติ / ไม่เป็นโรคหัวใจ")
    st.image("./img/katawut.jpeg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# โหลดข้อมูล Heart3.csv
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

# คำนวณค่าเฉลี่ยของตัวแปรเชิงตัวเลขหลักๆ เพื่อนำมาแสดงกราฟ (ใช้ค่าเฉลี่ยจะเห็นภาพชัดกว่าผลรวม)
dt1 = dt['Age'].mean()
dt2 = dt['Sex'].mean()
dt3 = dt['ChestPainType'].mean()
dt4 = dt['RestingBP'].mean()
dt5 = dt['Cholesterol'].mean()
dt6 = dt['FastingBS'].mean()
dt7 = dt['RestingECG'].mean()
dt8 = dt['MaxHR'].mean()
dt9 = dt['ExerciseAngina'].mean()
dt10 = dt['Oldpeak'].mean()
dt11 = dt['ST_Slope'].mean()

dx = [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11]
dx2 = pd.DataFrame(dx, index=["Age (อายุ)","Sex(เพศ)","ChestPainType(อาการเจ็บหน้าอก)" , "RestingBP (ความดัน)", "Cholesterol (ไขมัน)","Fasting(น้ำตาลในเลือด)","RestingECG(คลื่นไฟฟ้าหัวใจขณะพัก)" , "MaxHR (ชีพจรสูงสุด)", "ExerciseAngina(เจ็บหน้าอกขณะออกกำลังกาย)", "Oldpeak(ค่าความผิดปกติของคลื่นหัวใจ)", "ST_Slope(ความชันกราฟ)"], columns=["ค่าเฉลี่ย"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# ส่วนรับข้อมูล 11 ฟีเจอร์ให้ตรงกับคอลัมน์ใน Heart3.csv
age = st.slider("กรุณาเลือกข้อมูล อายุ (Age)", min_value=1, max_value=120, value=40)
sex = st.selectbox("กรุณาเลือกข้อมูล เพศ (Sex: 1 = ชาย, 0 = หญิง)", options=[1, 0])
cp_type = st.number_input("กรุณาเลือกข้อมูล ประเภทอาการเจ็บหน้าอก (ChestPainType: 1-4)", min_value=1, max_value=4, value=2)
r_bp = st.number_input("กรุณาเลือกข้อมูล ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=250, value=120)
Ch = st.number_input("กรุณาเลือกข้อมูล คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=200)
f_bs = st.selectbox("กรุณาเลือกข้อมูล น้ำตาลในเลือด (FastingBS: 1 = >120 mg/dl, 0 = ปกติ)", options=[0, 1])
r_ecg = st.number_input("กรุณาเลือกข้อมูล คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG: 1-3)", min_value=1, max_value=3, value=2)
m_hr = st.number_input("กรุณาเลือกข้อมูล อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=50, max_value=250, value=150)
ea = st.selectbox("กรุณาเลือกข้อมูล เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina: 1 = ใช่, 0 = ไม่)", options=[0, 1])
Old = st.number_input("กรุณาเลือกข้อมูล ค่าความผิดปกติของคลื่นหัวใจ (Oldpeak)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
st_s = st.number_input("กรุณาเลือกข้อมูล ความชันกราฟ ST (ST_Slope: 1-3)", min_value=1, max_value=3, value=2)

if st.button("ทำนายผล"):
    # แยก Feature (X) และ Target (y) โดยใช้คอลัมน์ HeartDisease
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']
    
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)  

    # เรียงลำดับตัวแปรให้ตรงกับคอลัมน์ใน X
    x_input = np.array([[age, sex, cp_type, r_bp, Ch, f_bs, r_ecg, m_hr, ea, Old, st_s]])

    out = Knn_model.predict(x_input)

    if out[0] == 1:
        st.error("ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (1)")
        st.image("./img/heart-attack.jpg")
    else:
        st.success("ผลการทำนาย: ปกติ / ไม่เป็นโรคหัวใจ (0)")
        st.image("./img/katawut.jpeg")
else:
    st.write("ไม่ทำนาย")