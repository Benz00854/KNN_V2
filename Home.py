from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บและตกแต่ง CSS ให้สวยงาม
# ---------------------------------------------------------
st.set_page_config(
    page_title="โปรเจคการจำแนกข้อมูลโรคหัวใจ (KNN)",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    .hero-banner {
        background: linear-gradient(135deg, #e11d48 0%, #881337 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(225, 29, 72, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #ffe4e6;
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }
    .section-header-red {
        background: linear-gradient(90deg, #EC7063 0%, #e11d48 100%);
        padding: 14px 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(236, 112, 99, 0.3);
        margin: 1.5rem 0 1rem 0;
    }
    .section-header-cyan {
        background: linear-gradient(90deg, #0ea5e9 0%, #6BD5DA 100%);
        padding: 14px 20px;
        border-radius: 12px;
        color: #0f172a;
        text-align: center;
        box-shadow: 0 4px 12px rgba(107, 213, 218, 0.35);
        margin: 1.5rem 0 1rem 0;
    }
    .img-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันช่วยแสดงรูปภาพอย่างปลอดภัย (ป้องกัน Error บน Streamlit.io)
def show_safe_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"ไม่พบไฟล์รูปภาพ: {path}")

# ---------------------------------------------------------
# 2. ส่วนหัวของเว็บและการ์ดเปรียบเทียบ 2 กลุ่ม
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🫀 โปรเจคการจำแนกข้อมูลโรคหัวใจ</h1>
    <p>ระบบวิเคราะห์ข้อมูลทางคลินิกและพยากรณ์ความเสี่ยงโรคหัวใจด้วยเทคนิค K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="img-card"><h4 style="color:#dc2626; margin-bottom:10px;">⚠️ กลุ่มเสี่ยง / เป็นโรคหัวใจ (1)</h4>', unsafe_allow_html=True)
    show_safe_image("./img/heart-attack.jpg")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="img-card"><h4 style="color:#16a34a; margin-bottom:10px;">✅ กลุ่มปกติ / ไม่เป็นโรคหัวใจ (0)</h4>', unsafe_allow_html=True)
    show_safe_image("./img/katawut.jpeg")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. ส่วนแสดงสถิติข้อมูลโรคหัวใจ
# ---------------------------------------------------------
st.markdown("""
<div class="section-header-red">
    <h4 style="margin:0; color:white;">📊 สถิติข้อมูลโรคหัวใจ (Heart3.csv)</h4>
</div>
""", unsafe_allow_html=True)

# โหลดข้อมูล Heart3.csv จาก path เดิม
@st.cache_data
def load_heart_data():
    return pd.read_csv("./data/Heart3.csv")

dt = load_heart_data()

# แสดงตัวเลขสรุปภาพรวมแบบ Metric
m1, m2, m3, m4 = st.columns(4)
m1.metric("จำนวนข้อมูลทั้งหมด", f"{len(dt):,} ราย")
m2.metric("กลุ่มเสี่ยงโรคหัวใจ (1)", f"{(dt['HeartDisease'] == 1).sum():,} ราย")
m3.metric("กลุ่มปกติ (0)", f"{(dt['HeartDisease'] == 0).sum():,} ราย")
m4.metric("อายุเฉลี่ยผู้ป่วย", f"{dt['Age'].mean():.1f} ปี")

with st.expander("📋 คลิกเพื่อดูตัวอย่างตารางข้อมูลดิบ (10 แถวแรก)", expanded=True):
    st.dataframe(dt.head(10), use_container_width=True)

# คำนวณค่าเฉลี่ยของตัวแปรทั้ง 11 ตัวตามโค้ดเดิม
dx = [
    dt['Age'].mean(), dt['Sex'].mean(), dt['ChestPainType'].mean(),
    dt['RestingBP'].mean(), dt['Cholesterol'].mean(), dt['FastingBS'].mean(),
    dt['RestingECG'].mean(), dt['MaxHR'].mean(), dt['ExerciseAngina'].mean(),
    dt['Oldpeak'].mean(), dt['ST_Slope'].mean()
]
dx2 = pd.DataFrame(
    dx,
    index=[
        "Age (อายุ)", "Sex (เพศ)", "ChestPainType (อาการเจ็บหน้าอก)",
        "RestingBP (ความดัน)", "Cholesterol (ไขมัน)", "FastingBS (น้ำตาลในเลือด)",
        "RestingECG (คลื่นไฟฟ้าหัวใจขณะพัก)", "MaxHR (ชีพจรสูงสุด)",
        "ExerciseAngina (เจ็บหน้าอกขณะออกกำลังกาย)", "Oldpeak (ค่าความผิดปกติคลื่นหัวใจ)",
        "ST_Slope (ความชันกราฟ)"
    ],
    columns=["ค่าเฉลี่ย"]
)

# จัดการปุ่มแสดง/ซ่อนกราฟด้วย Session State ให้กดใช้งานได้ลื่นไหล
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 3])
with btn_col1:
    if st.button("📈 แสดงการจินตทัศน์ข้อมูล", use_container_width=True, type="primary"):
        st.session_state.show_chart = True
with btn_col2:
    if st.button("🙈 ไม่แสดงข้อมูล", use_container_width=True):
        st.session_state.show_chart = False

if st.session_state.show_chart:
    st.bar_chart(dx2, color="#e11d48")
else:
    st.info("กดปุ่ม **'แสดงการจินตทัศน์ข้อมูล'** ด้านบนเพื่อดูกราฟค่าเฉลี่ยของแต่ละปัจจัย")

# ---------------------------------------------------------
# 4. ส่วนรับข้อมูลและทำนายผลด้วย KNN
# ---------------------------------------------------------
st.markdown("""
<div class="section-header-cyan">
    <h4 style="margin:0; color:#0f172a;">🩺 ทำนายข้อมูลโรคหัวใจ (KNN Prediction)</h4>
</div>
""", unsafe_allow_html=True)

# ปรับค่า K ได้จากแถบด้านข้าง (ค่าเริ่มต้น K=5 ให้ความแม่นยำสูงสุด)
with st.sidebar:
    st.header("⚙️ ตั้งค่าโมเดล KNN")
    k_neighbors = st.slider("เลือกจำนวนเพื่อนบ้าน (n_neighbors / K)", min_value=1, max_value=15, value=5, step=2)
    use_scaler = st.checkbox("ปรับมาตรฐานข้อมูล (StandardScaler)", value=True, help="ช่วยให้ KNN คำนวณระยะห่างระหว่างตัวแปรได้อย่างแม่นยำขึ้น")

# แบ่งช่องกรอกข้อมูล 11 ฟีเจอร์เป็น 3 คอลัมน์ให้สวยงามและกรอกง่าย
in_col1, in_col2, in_col3 = st.columns(3)

with in_col1:
    st.markdown("##### 👤 ข้อมูลทั่วไปและอาการ")
    age = st.slider("อายุ (Age)", min_value=1, max_value=120, value=40)
    sex = st.selectbox("เพศ (Sex)", options=[1, 0], format_func=lambda x: "1 = ชาย" if x == 1 else "0 = หญิง")
    cp_type = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก (ChestPainType)",
        options=[1, 2, 3, 4],
        index=1,
        format_func=lambda x: {
            1: "1 - ไม่มีอาการ (ASY)",
            2: "2 - เจ็บแบบไม่ตรงเกณฑ์ (ATA)",
            3: "3 - เจ็บที่ไม่ใช่จากหัวใจ (NAP)",
            4: "4 - เจ็บแบบตรงเกณฑ์ (TA)"
        }[x]
    )
    ea = st.selectbox(
        "เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
        options=[0, 1],
        format_func=lambda x: "1 = ใช่ (มีอาการ)" if x == 1 else "0 = ไม่ (ไม่มีอาการ)"
    )

with in_col2:
    st.markdown("##### 🩸 ความดันและผลตรวจเลือด")
    r_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=250, value=120)
    Ch = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=200)
    f_bs = st.selectbox(
        "น้ำตาลในเลือด (FastingBS)",
        options=[0, 1],
        format_func=lambda x: "1 = > 120 mg/dl" if x == 1 else "0 = ปกติ (<= 120 mg/dl)"
    )
    m_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=50, max_value=250, value=150)

with in_col3:
    st.markdown("##### 📈 ผลคลื่นไฟฟ้าหัวใจ (ECG)")
    r_ecg = st.selectbox(
        "คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: {
            1: "1 - ผิดปกติ (LVH)",
            2: "2 - ปกติ (Normal)",
            3: "3 - คลื่น ST-T ผิดปกติ (ST)"
        }[x]
    )
    Old = st.number_input("ค่าความผิดปกติของคลื่นหัวใจ (Oldpeak)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
    st_s = st.selectbox(
        "ความชันกราฟ ST (ST_Slope)",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: {
            1: "1 - ความชันลดลง (Down)",
            2: "2 - ความชันแบนราบ (Flat)",
            3: "3 - ความชันเพิ่มขึ้น (Up)"
        }[x]
    )

st.write("")
if st.button("🔍 ทำนายผล", use_container_width=True, type="primary"):
    # แยก Feature (X) และ Target (y)
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']
    
    x_input = pd.DataFrame(
        [[age, sex, cp_type, r_bp, Ch, f_bs, r_ecg, m_hr, ea, Old, st_s]],
        columns=X.columns
    )

    if use_scaler:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        x_input_processed = scaler.transform(x_input)
    else:
        X_processed = X
        x_input_processed = x_input

    Knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    Knn_model.fit(X_processed, y)

    out = Knn_model.predict(x_input_processed)
    prob = Knn_model.predict_proba(x_input_processed)[0]

    st.divider()
    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if out[0] == 1:
            st.error(f"### ⚠️ ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (1)\nความน่าจะเป็นที่คำนวณได้: **{prob[1]*100:.1f}%** (ใช้ค่า K = {k_neighbors})")
        else:
            st.success(f"### ✅ ผลการทำนาย: ปกติ / ไม่เป็นโรคหัวใจ (0)\nความน่าจะเป็นที่ปกติ: **{prob[0]*100:.1f}%** (ความเสี่ยงโรคหัวใจ {prob[1]*100:.1f}%, ค่า K = {k_neighbors})")

    with res_col2:
        if out[0] == 1:
            show_safe_image("./img/heart-attack.jpg", caption="กลุ่มเสี่ยง / เป็นโรคหัวใจ")
        else:
            show_safe_image("./img/katawut.jpeg", caption="กลุ่มปกติ / ไม่เป็นโรคหัวใจ")
else:
    st.caption("กรอกข้อมูลด้านบนให้ครบถ้วน แล้วกดปุ่ม **'ทำนายผล'** เพื่อดูผลลัพธ์")