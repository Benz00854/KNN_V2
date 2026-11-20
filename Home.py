from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บและตกแต่ง CSS สไตล์ Kawaii Pastel ✨💖
# ---------------------------------------------------------
st.set_page_config(
    page_title="💖 CardioKawaii | เช็กสุขภาพหัวใจสุดคิวท์",
    page_icon="💖",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mali:wght@400;500;600;700&family=Prompt:wght@300;400;500;600&display=swap');
    
    /* พื้นหลังโทนชมพูครีมพาสเทลอ่อนๆ */
    .stApp {
        background: linear-gradient(180deg, #fff5f8 0%, #fff0f5 50%, #fdf4ff 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Mali', cursive !important;
    }

    /* แบนเนอร์หัวเว็บสุดคิวท์ */
    .kawaii-banner {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        padding: 2.2rem;
        border-radius: 28px;
        border: 3px dashed #ffffff;
        color: #881337;
        text-align: center;
        box-shadow: 0 12px 25px rgba(255, 154, 158, 0.35);
        margin-bottom: 1.8rem;
    }
    .kawaii-banner h1 {
        color: #9f1239;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        text-shadow: 2px 2px 0px #ffffff;
    }
    .kawaii-banner p {
        color: #be123c;
        margin-top: 0.6rem;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* หัวข้อแบ่งส่วนแบบเมฆพาสเทล */
    .kawaii-header-pink {
        background: linear-gradient(90deg, #ff758c 0%, #ff7eb3 100%);
        padding: 14px 24px;
        border-radius: 50px;
        border: 2px solid #ffffff;
        color: white;
        text-align: center;
        box-shadow: 0 6px 15px rgba(255, 117, 140, 0.35);
        margin: 1.8rem 0 1.2rem 0;
    }
    .kawaii-header-mint {
        background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 14px 24px;
        border-radius: 50px;
        border: 2px solid #ffffff;
        color: #1e3a8a;
        text-align: center;
        box-shadow: 0 6px 15px rgba(161, 196, 253, 0.4);
        margin: 1.8rem 0 1.2rem 0;
    }

    /* การ์ดรูปภาพขอบมนเหมือนโพลารอยด์ */
    .kawaii-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 24px;
        border: 3px solid #fbcfe8;
        box-shadow: 0 8px 20px rgba(244, 114, 182, 0.15);
        text-align: center;
    }
    
    /* ตกแต่งกล่อง Metric */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #fbcfe8;
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(244, 114, 182, 0.12);
    }

    /* ตกแต่งปุ่มกดให้กลมมนเหมือนขนมมาการอง */
    .stButton > button {
        border-radius: 50px !important;
        border: 2px solid #ffffff !important;
        font-family: 'Mali', cursive !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(244, 114, 182, 0.25) !important;
        transition: transform 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.03) !important;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันแสดงรูปภาพอย่างปลอดภัย
def show_safe_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"🥺 เอ๊ะ! หาไฟล์รูปไม่เจอ: {path}")

# ---------------------------------------------------------
# 2. ส่วนหัวของเว็บและการ์ดเปรียบเทียบ 2 กลุ่ม 🌸
# ---------------------------------------------------------
st.markdown("""
<div class="kawaii-banner">
    <h1>💖 โปรเจคการจำแนกข้อมูลโรคหัวใจ ✨</h1>
    <p>🩺 น้อง AI ช่วยวิเคราะห์ความเสี่ยงโรคหัวใจด้วยเทคนิค K-Nearest Neighbors (KNN) 🌸</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="kawaii-card"><h4 style="color:#e11d48; margin-bottom:12px;">💔 กลุ่มเสี่ยง / เป็นโรคหัวใจ (1)</h4>', unsafe_allow_html=True)
    show_safe_image("./img/heart-attack.jpg")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kawaii-card"><h4 style="color:#059669; margin-bottom:12px;">💚 กลุ่มปกติ / หัวใจแข็งแรง (0)</h4>', unsafe_allow_html=True)
    show_safe_image("./img/katawut.jpeg")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. ส่วนแสดงสถิติข้อมูลโรคหัวใจ 🎀
# ---------------------------------------------------------
st.markdown("""
<div class="kawaii-header-pink">
    <h4 style="margin:0; color:white;">🎀 สถิติข้อมูลโรคหัวใจสุดคิวท์ (Heart3.csv) 🎀</h4>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_heart_data():
    return pd.read_csv("./data/Heart3.csv")

dt = load_heart_data()

# แสดงการ์ดตัวเลขสรุปน่ารักๆ
m1, m2, m3, m4 = st.columns(4)
m1.metric("🧸 ข้อมูลทั้งหมด", f"{len(dt):,} ราย")
m2.metric("💔 กลุ่มเสี่ยงโรคหัวใจ", f"{(dt['HeartDisease'] == 1).sum():,} ราย")
m3.metric("💚 กลุ่มหัวใจแข็งแรง", f"{(dt['HeartDisease'] == 0).sum():,} ราย")
m4.metric("🎂 อายุเฉลี่ย", f"{dt['Age'].mean():.1f} ปี")

st.write("")
with st.expander("📖 คลิกเพื่อแอบดูตารางข้อมูลดิบ (10 แถวแรก) ✨", expanded=True):
    st.dataframe(dt.head(10), use_container_width=True)

# คำนวณค่าเฉลี่ยของตัวแปรทั้ง 11 ตัว
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

if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

btn_col1, btn_col2, _ = st.columns([1.3, 1.3, 3])
with btn_col1:
    if st.button("🌈 แสดงการจินตทัศน์ข้อมูล", use_container_width=True, type="primary"):
        st.session_state.show_chart = True
with btn_col2:
    if st.button("🙈 ซ่อนกราฟไว้ก่อน", use_container_width=True):
        st.session_state.show_chart = False

if st.session_state.show_chart:
    st.bar_chart(dx2, color="#ff758c")
else:
    st.info("✨ กดปุ่ม **'🌈 แสดงการจินตทัศน์ข้อมูล'** ด้านบนเพื่อดูกราฟค่าเฉลี่ยแสนน่ารักได้เลย!")

# ---------------------------------------------------------
# 4. ส่วนรับข้อมูลและทำนายผลด้วย KNN 🩺✨
# ---------------------------------------------------------
st.markdown("""
<div class="kawaii-header-mint">
    <h4 style="margin:0; color:#1e3a8a;">🩺 มาทำนายสุขภาพหัวใจกันเถอะ! (KNN Prediction) ✨</h4>
</div>
""", unsafe_allow_html=True)

# แถบด้านข้างปรับแต่งโมเดล
with st.sidebar:
    st.markdown("### 🧸 ตั้งค่าน้องโมเดล KNN")
    k_neighbors = st.slider("จำนวนเพื่อนบ้านใกล้เคียง (K) 🌸", min_value=1, max_value=15, value=5, step=2)
    use_scaler = st.checkbox("✨ ปรับมาตรฐานข้อมูล (StandardScaler)", value=True)
    st.caption("💡 แนะนำให้เปิดไว้ เพื่อให้น้อง KNN คำนวณแม่นยำขึ้นนะ!")

# แบ่งช่องกรอกข้อมูล 3 คอลัมน์
in_col1, in_col2, in_col3 = st.columns(3)

with in_col1:
    st.markdown("##### 🐰 ข้อมูลทั่วไปและอาการ")
    age = st.slider("อายุ (Age) 🎂", min_value=1, max_value=120, value=40)
    sex = st.selectbox("เพศ (Sex) 🚻", options=[1, 0], format_func=lambda x: "1 = ชาย 👦" if x == 1 else "0 = หญิง 👧")
    cp_type = st.selectbox(
        "อาการเจ็บหน้าอก (ChestPainType) 💓",
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
        "เจ็บหน้าอกตอนออกกำลังกาย (ExerciseAngina) 🏃‍♀️",
        options=[0, 1],
        format_func=lambda x: "1 = ใช่ (มีอาการ) 😣" if x == 1 else "0 = ไม่ (สบายดี) 😊"
    )

with in_col2:
    st.markdown("##### 🍓 ความดันและผลตรวจเลือด")
    r_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP) 🩺", min_value=0, max_value=250, value=120)
    Ch = st.number_input("คอเลสเตอรอล (Cholesterol) 🍔", min_value=0, max_value=700, value=200)
    f_bs = st.selectbox(
        "น้ำตาลในเลือด (FastingBS) 🍬",
        options=[0, 1],
        format_func=lambda x: "1 = > 120 mg/dl 🍭" if x == 1 else "0 = ปกติ (<= 120 mg/dl) 🥗"
    )
    m_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR) 💓", min_value=50, max_value=250, value=150)

with in_col3:
    st.markdown("##### 🌸 ผลคลื่นไฟฟ้าหัวใจ (ECG)")
    r_ecg = st.selectbox(
        "คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG) 📈",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: {
            1: "1 - ผิดปกติ (LVH)",
            2: "2 - ปกติ (Normal)",
            3: "3 - คลื่น ST-T ผิดปกติ (ST)"
        }[x]
    )
    Old = st.number_input("ค่าความผิดปกติของคลื่นหัวใจ (Oldpeak) 📉", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
    st_s = st.selectbox(
        "ความชันกราฟ ST (ST_Slope) 📐",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: {
            1: "1 - ความชันลดลง (Down)",
            2: "2 - ความชันแบนราบ (Flat)",
            3: "3 - ความชันเพิ่มขึ้น (Up)"
        }[x]
    )

st.write("")
if st.button("💖 กดเพื่อทำนายผลเลย! ✨", use_container_width=True, type="primary"):
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
            st.error(f"### 🥺 ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (1)\nโอกาสเสี่ยงที่น้อง AI คำนวณได้คือ **{prob[1]*100:.1f}%** (อ้างอิงจากเพื่อนบ้าน K = {k_neighbors})\n\n💌 *ดูแลสุขภาพด้วยน้า แนะนำให้ลองปรึกษาคุณหมอเพื่อตรวจเช็กอย่างละเอียดนะครับ!*")
        else:
            st.balloons()
            st.success(f"### 🎉 เย้! ผลการทำนาย: ปกติ / หัวใจแข็งแรง (0)\nความน่าจะเป็นที่หัวใจปกติสูงถึง **{prob[0]*100:.1f}%** (ความเสี่ยงเพียง {prob[1]*100:.1f}%, ค่า K = {k_neighbors})\n\n🌟 *สุขภาพดีมากเลย! รักษาสุขภาพและออกกำลังกายสม่ำเสมอต่อไปนะครับ!*")

    with res_col2:
        if out[0] == 1:
            show_safe_image("./img/heart-attack.jpg", caption="💔 กลุ่มเสี่ยง / เป็นโรคหัวใจ")
        else:
            show_safe_image("./img/katawut.jpeg", caption="💚 กลุ่มปกติ / หัวใจแข็งแรง")
else:
    st.caption("🌸 กรอกข้อมูลด้านบนให้ครบ แล้วกดปุ่ม **'💖 กดเพื่อทำนายผลเลย! ✨'** เพื่อดูผลลัพธ์น้า~")