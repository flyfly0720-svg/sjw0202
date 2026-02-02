import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(layout="wide")
st.title("🧠 설득·재유입 확장 SIR 정보 확산 모델")

st.markdown("""
이 모델은  
- **설득 계수 θ (정보 신뢰도)**  
- **재유입률 ρ (R → S)**  
를 추가한 확장 SIR 모델입니다.
""")

# -----------------------------
# 파라미터 입력
# -----------------------------
st.sidebar.header("⚙️ 모델 파라미터")

beta = st.sidebar.slider("기본 확산률 β", 0.05, 1.0, 0.3)
gamma = st.sidebar.slider("망각률 γ", 0.05, 1.0, 0.2)
theta = st.sidebar.slider("설득 계수 θ", 0.1, 2.0, 1.0)
rho = st.sidebar.slider("재유입률 ρ (R → S)", 0.0, 0.5, 0.05)

days = st.sidebar.slider("시뮬레이션 기간 (일)", 30, 365, 120)

# -----------------------------
# 초기 상태
# -----------------------------
S0, I0, R0 = 0.95, 0.05, 0.0
dt = 0.1
t = np.arange(0, days, dt)

S = np.zeros(len(t))
I = np.zeros(len(t))
R = np.zeros(len(t))

S[0], I[0], R[0] = S0, I0, R0

# -----------------------------
# Euler 수치해석
# -----------------------------
for i in range(1, len(t)):
    dS = -beta * theta * S[i-1] * I[i-1] + rho * R[i-1]
    dI = beta * theta * S[i-1] * I[i-1] - gamma * I[i-1]
    dR = gamma * I[i-1] - rho * R[i-1]

    S[i] = S[i-1] + dS * dt
    I[i] = I[i-1] + dI * dt
    R[i] = R[i-1] + dR * dt

# -----------------------------
# 데이터프레임
# -----------------------------
df = pd.DataFrame({
    "S (잠재 수용자)": S,
    "I (정보 확산자)": I,
    "R (망각 집단)": R
}, index=t)

df.index.name = "Time"

# -----------------------------
# 시각화 (Streamlit 내장)
# -----------------------------
st.subheader("📈 정보 확산 동역학")
st.line_chart(df)

# -----------------------------
# 핵심 지표
# -----------------------------
st.subheader("🔍 핵심 지표 요약")

col1, col2, col3 = st.columns(3)

col1.metric("최대 확산 비율 (I max)", f"{I.max():.2f}")
col2.metric("최대 확산 시점 (일)", f"{t[I.argmax()]:.1f}")
col3.metric("장기 잔존 비율 (R)", f"{R[-1]:.2f}")

# -----------------------------
# 해석 가이드
# -----------------------------
st.subheader("🧠 해석 가이드")

st.markdown("""
- **θ 증가** → 정보가 사실이 아니어도 *그럴듯하면* 급확산  
- **ρ > 0** → 정보는 완전히 사라지지 않고 **사회에 재순환**  
- 이 구조는  
  - 가짜뉴스  
  - 밈  
  - 유행어  
  - 건강 정보  
에 그대로 적용 가능
""")
