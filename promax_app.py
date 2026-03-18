import streamlit as st
import pandas as pd
import hashlib
import os
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ===== CONFIG =====
APP_NAME = "Promax v4 🔥"
st.set_page_config(page_title="Promax", layout="wide")

# ===== DATABASE =====
USERS_DB = "users.csv"
DATA_DB = "data.csv"

if not os.path.exists(USERS_DB):
    pd.DataFrame(columns=["Username","Password"]).to_csv(USERS_DB,index=False)

if not os.path.exists(DATA_DB):
    pd.DataFrame().to_csv(DATA_DB,index=False)

# ===== MODELS =====
models={
    "PCX160":{"base_factor":1.5,"max_speed":150},
    "Click160":{"base_factor":1.4,"max_speed":145},
    "ADV160":{"base_factor":1.3,"max_speed":145},
    "Scoopy":{"base_factor":1.2,"max_speed":120}
}

# ===== AUTH =====
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(u,p):
    df=pd.read_csv(USERS_DB)
    if u in df["Username"].values: return False
    pd.DataFrame([{"Username":u,"Password":hash_pw(p)}]).to_csv(USERS_DB,mode='a',header=False,index=False)
    return True

def login(u,p):
    df=pd.read_csv(USERS_DB)
    user=df[df["Username"]==u]
    return not user.empty and user["Password"].values[0]==hash_pw(p)

# ===== CORE ENGINE =====
def advanced_predict(
    model, weights, current_speed=None, rpm=None,
    fuel="95", gear_ratio=1.0, air_filter=False,
    remap=False, exhaust=False, wind=False,
    rider="normal", distance_km=None, tuck=False, engine_type="stock"
):
    base=models[model]["base_factor"]
    max_speed=models[model]["max_speed"]

    avg_weight=sum(weights)/len(weights)
    balance=max(weights)-min(weights)

    factor=base*(1-balance*0.02)
    delta=(15-avg_weight)

    fuel_map={"91":0.97,"95":1.0,"E20":1.03,"E85":1.07}
    delta*=fuel_map.get(fuel,1.0)

    if remap: delta*=1.2
    if exhaust: delta*=1.1
    if air_filter: delta*=1.05
    if wind: delta*=0.9

    if rider=="light": delta*=1.05
    elif rider=="heavy": delta*=0.95

    if engine_type=="big": delta*=1.1
    if rpm: delta*=(rpm/8000)

    speed=(delta*factor*gear_ratio)

    if tuck: speed*=1.05

    if current_speed is None:
        current_speed=max_speed*0.85

    final_speed=min(current_speed+speed,max_speed)

    time_sec=None
    if distance_km:
        time_sec=(distance_km*1000)/(final_speed*1000/3600)

    return round(final_speed,1), time_sec

# ===== AI =====
def train_ai():
    if not os.path.exists(DATA_DB): return None
    df=pd.read_csv(DATA_DB)
    if len(df)<5: return None
    X=df[["avg_weight","gear_ratio"]]
    y=df["speed"]
    model=LinearRegression().fit(X,y)
    return model

def ai_predict(model,avg,gear):
    return round(model.predict([[avg,gear]])[0],1)

# ===== UI =====
st.title(APP_NAME)

menu=st.sidebar.radio("Menu",["Login","Register"])

if menu=="Register":
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")
    if st.button("Register"):
        st.success("สมัครสำเร็จ") if register(u,p) else st.error("มี user นี้แล้ว")

elif menu=="Login":
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")
    if st.button("Login"):
        if login(u,p):
            st.session_state["user"]=u
            st.success("เข้าแล้ว")
        else:
            st.error("ผิด")

# ===== DASHBOARD =====
if "user" in st.session_state:
    st.subheader(f"Welcome {st.session_state['user']}")

    col1,col2=st.columns(2)

    with col1:
        model=st.selectbox("รุ่น",list(models.keys()))
        num=st.selectbox("จำนวนเม็ด",[4,5,6])

        weights=[]
        cols=st.columns(num)
        for i in range(num):
            weights.append(cols[i].number_input(f"เม็ด{i+1}",8.0,20.0,15.0,key=f"w{i}"))

        use_speed=st.checkbox("ใส่ความเร็ว")
        speed=None
        if use_speed:
            speed=st.number_input("km/h",0.0,200.0,130.0)

    with col2:
        fuel=st.selectbox("น้ำมัน",["95","91","E20","E85"])
        rpm_use=st.checkbox("ใส่รอบ")
        rpm=None
        if rpm_use:
            rpm=st.number_input("RPM",1000,15000,8000)

        gear=st.slider("เฟือง",0.8,1.3,1.0)
        air=st.checkbox("กรองเลส")
        remap=st.checkbox("รีแมพ")
        exhaust=st.checkbox("ท่อ")
        wind=st.checkbox("ลม")
        tuck=st.checkbox("หมอบ")

        rider=st.selectbox("น้ำหนักคน",["normal","light","heavy"])
        engine=st.selectbox("ลูก",["stock","big"])

        use_dist=st.checkbox("คำนวณระยะ")
        dist=None
        if use_dist:
            dist=st.number_input("กม.",0.1,10.0,2.0)

    if st.button("🚀 คำนวณ"):
        speed_out,time_out=advanced_predict(
            model,weights,speed,rpm,fuel,gear,air,
            remap,exhaust,wind,rider,dist,tuck,engine
        )

        st.success(f"ความเร็ว: {speed_out} km/h")

        if time_out:
            st.info(f"เวลา: {time_out:.2f} วิ")

        # save
        avg=sum(weights)/len(weights)
        pd.DataFrame([{
            "user":st.session_state["user"],
            "avg_weight":avg,
            "gear_ratio":gear,
            "speed":speed_out
        }]).to_csv(DATA_DB,mode='a',header=not os.path.exists(DATA_DB),index=False)

        # AI
        ai=train_ai()
        if ai:
            ai_s=ai_predict(ai,avg,gear)
            st.warning(f"🤖 AI: {ai_s} km/h")

    if os.path.exists(DATA_DB):
        st.subheader("History")
        st.dataframe(pd.read_csv(DATA_DB))