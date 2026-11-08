"""
Recurso interactivo - Escenario 4 (Ciencia de Datos), Situación 4A: Predicción de demanda
Cálculo diferencial 2026-II · Tercer corte

Cómo correrlo:
    pip install streamlit plotly numpy pandas sympy
    streamlit run app.py

Unidades: V y V* están en CIENTOS de visitas diarias (x100 = visitas). t son días desde que
empezó la campaña. Dominio de los modelos: el primer mes, 0 <= t <= 30 (ver pestaña "Dominio").
"""
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from plotly.subplots import make_subplots

st.set_page_config(page_title="Predicción de demanda · Situación 4A", layout="wide")

def es(x, d=0, signo=""):
    """Formato español: punto para miles y coma para decimales (1.518,75)."""
    texto = f"{float(x):{signo},.{d}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# DATOS DEL GRUPO (se cambian aquí o en la barra lateral)
# ---------------------------------------------------------------------------
K_GRUPO_POR_DEFECTO = 4          # <- número de grupo que asigna el docente
DATOS_BASE = [120, 150, 190, 240, 300, 360]   # los seis primeros datos de la Pregunta 5
TMAX = 30                        # el análisis es del primer mes
T_CAMPANA = 5                    # la campaña dura 5 días

AZUL, TEAL, NARANJA, ROJO, GRIS = "#1E2761", "#028090", "#E07A1F", "#C0392B", "#6B7280"


# ---------------------------------------------------------------------------
# MODELOS (cientos de visitas diarias) Y SUS DERIVADAS (cientos de visitas por día cada día)
# ---------------------------------------------------------------------------
def v_exp(t):
    """Modelo exponencial de la campaña: 200*(1,5)^t, válido para 0 <= t <= 5."""
    return 200 * 1.5 ** t


def dv_exp(t):
    """Derivada de 200*(1,5)^t  ->  200*ln(1,5)*(1,5)^t"""
    return 200 * np.log(1.5) * 1.5 ** t


def v_post(t, v0=900.0, r=0.8):
    """Regla después de la campaña: v0*r^(t-5), válida para 5 < t <= 30."""
    return v0 * r ** (t - T_CAMPANA)


def dv_post(t, v0=900.0, r=0.8):
    """Derivada: v0*ln(r)*r^(t-5)"""
    return v0 * np.log(r) * r ** (t - T_CAMPANA)


def v_trozos(t, v0=900.0, r=0.8):
    t = np.asarray(t, dtype=float)
    return np.where(t <= T_CAMPANA, v_exp(t), v_post(t, v0, r))


def dv_trozos(t, v0=900.0, r=0.8):
    t = np.asarray(t, dtype=float)
    return np.where(t <= T_CAMPANA, dv_exp(t), dv_post(t, v0, r))


def v_cor(t, K=2000.0, a=9.0, b=0.6):
    """Modelo corregido V*(t) = K / (1 + a*b^t)."""
    return K / (1 + a * b ** t)


def dv_cor(t, K=2000.0, a=9.0, b=0.6):
    """Derivada: V*'(t) = -K*a*ln(b)*b^t / (1 + a*b^t)^2  (positiva porque ln b < 0)"""
    u = b ** t
    return -K * a * np.log(b) * u / (1 + a * u) ** 2


def d2v_cor(t, K=2000.0, a=9.0, b=0.6):
    """Segunda derivada: -K*a*(ln b)^2 * b^t * (1 - a*b^t) / (1 + a*b^t)^3"""
    u = b ** t
    return -K * a * np.log(b) ** 2 * u * (1 - a * u) / (1 + a * u) ** 3


def t_inflexion(a=9.0, b=0.6):
    """V*'' = 0 cuando a*b^t = 1  ->  t = ln(a) / (-ln b)"""
    return np.log(a) / (-np.log(b))


def t_para_nivel(C, K, a, b):
    """Día en que V* llega al nivel C (cientos). None si C no se alcanza."""
    if C <= K / (1 + a) or C >= K:
        return None
    return float(np.log((K / C - 1) / a) / np.log(b))


# ---------------------------------------------------------------------------
# BARRA LATERAL: lo que la organización puede mover
# ---------------------------------------------------------------------------
ESCENARIOS = {
    "Base (los datos del escenario)": dict(tope=200_000, a=9.0, b=0.6, v0=900, r=0.8),
    "¿Y si el tope fuera mayor? (250.000)": dict(tope=250_000, a=9.0, b=0.6, v0=900, r=0.8),
    "¿Y si el crecimiento fuera más lento? (b = 0,7)": dict(tope=200_000, a=9.0, b=0.7, v0=900, r=0.8),
    "¿Y si fuera más rápido? (b = 0,5)": dict(tope=200_000, a=9.0, b=0.5, v0=900, r=0.8),
    "¿Y si no hubiera caída al terminar la campaña? (900 → 1.518,75)": dict(tope=200_000, a=9.0, b=0.6, v0=1518.75, r=0.8),
}


def aplicar_escenario():
    e = ESCENARIOS[st.session_state["escenario"]]
    st.session_state["tope"] = e["tope"]
    st.session_state["a"] = e["a"]
    st.session_state["b"] = e["b"]
    st.session_state["v0"] = float(e["v0"])
    st.session_state["r"] = e["r"]


for clave, valor in dict(tope=200_000, a=9.0, b=0.6, v0=900.0, r=0.8, capacidad=150_000).items():
    st.session_state.setdefault(clave, valor)

with st.sidebar:
    st.header("Datos que puede cambiar")
    k = st.number_input("Número de grupo (k)", 1, 99, K_GRUPO_POR_DEFECTO, 1,
                        help="El docente lo asigna. Solo cambia el dato 430+10k de la Pregunta 5.")
    st.selectbox("Escenario «¿qué pasa si…?»", list(ESCENARIOS), key="escenario", on_change=aplicar_escenario)
    st.markdown("**Modelo corregido V\\***")
    tope = st.slider("Tope de visitas diarias (visitas)", 100_000, 400_000, step=10_000, key="tope", format="%d")
    a = st.slider("Parámetro a (qué tan abajo arranca)", 1.0, 20.0, step=0.5, key="a", format="%.1f",
                  help="V*(0) = tope/(1+a). Con a = 9 y tope 200.000 arranca en 20.000 visitas.")
    b = st.slider("Parámetro b (qué tan rápido se frena, entre 0 y 1)", 0.30, 0.90, step=0.05, key="b", format="%.2f",
                  help="Más pequeño = llega más rápido al tope.")
    st.markdown("**Después de la campaña (día 5)**")
    v0 = st.slider("Valor justo después del día 5 (cientos)", 300.0, 1600.0, step=10.0, key="v0", format="%.0f")
    r = st.slider("Qué tanto baja cada día (base)", 0.5, 0.95, step=0.05, key="r")
    st.markdown("**Servidores**")
    capacidad = st.number_input("Capacidad actual (visitas por día)", 20_000, 400_000, step=5_000, key="capacidad")
    anticipacion = st.slider("Días que tarda ampliar la infraestructura", 0.0, 3.0, 1.0, 0.5)
    margen = st.slider("Margen de seguridad (%)", 0, 30, 10, 5,
                       help="El 10 % es una decisión de prudencia del grupo, no sale de los cálculos.")

K = tope / 100.0                      # tope en cientos
t = np.linspace(0, TMAX, 601)
t_star = t_inflexion(a, b)
v_star_at = float(v_cor(t_star, K, a, b))
rate_max = float(dv_cor(t_star, K, a, b))
C_cientos = capacidad / 100.0
t_cap = t_para_nivel(C_cientos, K, a, b)

# ---------------------------------------------------------------------------
st.title("Predicción de demanda: visitas diarias del primer mes")
st.caption("Situación 4A · Ciencia de Datos · Cálculo diferencial 2026-II. "
           "Las visitas están en cientos (x100 = visitas). t = días desde que empezó la campaña.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 · Visitas del mes", "2 · Campaña y cambio de regla", "3 · Un valor para resumir la semana",
     "4 · Decisión", "Fórmulas, dominio y verificación"])

# ============================ TAB 1 =========================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Día en que más rápido crecen", f"día {es(t_star,2,'')}")
    c2.metric("Visitas diarias ese día", f"{es(v_star_at*100,0,'')}")
    c3.metric("Ritmo máximo (visitas/día, cada día)", f"+{es(rate_max*100,0,'')}")
    c4.metric("Visitas diarias el día 30", f"{es(float(v_cor(30, K, a, b))*100,0,'')}",
              delta=f"{es((float(v_cor(30, K, a, b))-K)*100,1,'')} del tope",
              delta_color="off")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Modelo corregido V*(t): visitas diarias", "Su ritmo de cambio V*′(t)"))
    fig.add_trace(go.Scatter(x=t, y=v_cor(t, K, a, b), name="V*(t)", line=dict(color=TEAL, width=3)), 1, 1)
    fig.add_hline(y=K, line=dict(color=GRIS, dash="dash"), row=1, col=1,
                  annotation_text=f"Tope: {es(K,0,'')} cientos = {es(tope,0,'')} visitas")
    fig.add_hline(y=C_cientos, line=dict(color=ROJO, dash="dot"), row=1, col=1,
                  annotation_text=f"Capacidad actual: {es(capacidad,0,'')}", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=[t_star], y=[v_star_at], mode="markers", name="Punto de inflexión",
                             marker=dict(color=NARANJA, size=12, symbol="diamond")), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dv_cor(t, K, a, b), name="V*′(t)", line=dict(color=NARANJA, width=3)), 2, 1)
    fig.add_trace(go.Scatter(x=[t_star], y=[rate_max], mode="markers", showlegend=False,
                             marker=dict(color=NARANJA, size=12, symbol="diamond")), 2, 1)
    if t_cap is not None and t_cap <= TMAX:
        fig.add_trace(go.Scatter(x=[t_cap], y=[C_cientos], mode="markers", name="Se llega a la capacidad",
                                 marker=dict(color=ROJO, size=11)), 1, 1)
    fig.add_vline(x=t_star, line=dict(color=NARANJA, dash="dot"))
    fig.update_xaxes(title_text="t (días desde el inicio de la campaña)", range=[0, TMAX], row=2, col=1)
    fig.update_yaxes(title_text="Visitas diarias (cientos)", row=1, col=1)
    fig.update_yaxes(title_text="Cientos de visitas por día, cada día", row=2, col=1)
    fig.update_layout(height=620, legend=dict(orientation="h", y=1.1), margin=dict(t=70, b=30))
    st.plotly_chart(fig, width="stretch")

    dia = st.slider("Elija un día para leer el valor y el ritmo", 0.0, 30.0, 4.0, 0.1)
    cc1, cc2 = st.columns(2)
    cc1.info(f"**Día {es(dia,1,'')}:** {es(float(v_cor(dia, K, a, b))*100,0,'')} visitas diarias")
    cc2.info(f"**Ritmo:** {es(float(dv_cor(dia, K, a, b))*100,0,'')} visitas más por día cada día")

    sentido = "acelerando" if d2v_cor(dia, K, a, b) > 0 else "frenando"
    st.caption(f"En el día {es(dia,1,'')} el crecimiento está **{sentido}**. "
               f"El cambio de «acelerar» a «frenar» ocurre en el día {es(t_star,2,'')}.")

# ============================ TAB 2 =========================================
with tab2:
    st.markdown("**Primeros 30 días con las dos reglas** (la campaña dura 5 días; después las visitas bajan).")
    tt = np.linspace(0, TMAX, 1201)
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("V(t): visitas diarias con las dos reglas", "Su ritmo de cambio V′(t)"))
    izq, der = tt[tt <= T_CAMPANA], tt[tt > T_CAMPANA]
    fig2.add_trace(go.Scatter(x=izq, y=v_exp(izq), name="Durante la campaña", line=dict(color=TEAL, width=3)), 1, 1)
    fig2.add_trace(go.Scatter(x=der, y=v_post(der, v0, r), name="Después de la campaña", line=dict(color=AZUL, width=3)), 1, 1)
    fig2.add_trace(go.Scatter(x=[5], y=[float(v_exp(5))], mode="markers", showlegend=False,
                              marker=dict(color=TEAL, size=11)), 1, 1)
    fig2.add_trace(go.Scatter(x=[5], y=[v0], mode="markers", showlegend=False,
                              marker=dict(color="white", line=dict(color=AZUL, width=2), size=11)), 1, 1)
    fig2.add_trace(go.Scatter(x=izq, y=dv_exp(izq), name="V′ durante", line=dict(color=NARANJA, width=3)), 2, 1)
    fig2.add_trace(go.Scatter(x=der, y=dv_post(der, v0, r), name="V′ después", line=dict(color=ROJO, width=3)), 2, 1)
    fig2.add_vline(x=5, line=dict(color=GRIS, dash="dot"))
    fig2.update_xaxes(title_text="t (días)", range=[0, TMAX], row=2, col=1)
    fig2.update_yaxes(title_text="Visitas diarias (cientos)", row=1, col=1)
    fig2.update_yaxes(title_text="Cientos de visitas por día, cada día", row=2, col=1)
    fig2.update_layout(height=620, legend=dict(orientation="h", y=1.1), margin=dict(t=70, b=30))
    st.plotly_chart(fig2, width="stretch")

    izq5, der5 = float(v_exp(5)), float(v0)
    salto = der5 - izq5
    d_izq, d_der = float(dv_exp(5)), float(dv_post(5, v0, r))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ritmo el día 3 (visitas/día, cada día)", f"+{es(float(dv_exp(3))*100,0,'')}")
    prom = (float(v_exp(4)) - float(v_exp(2))) / 2
    m2.metric("Promedio días 2 a 4 (visitas/día)", f"+{es(prom*100,0,'')}",
              help="Es mayor que el ritmo exacto porque la curva se va empinando.")
    m3.metric("Ritmo el día 6 (visitas/día, cada día)", f"{es(float(dv_post(6, v0, r))*100,0,'')}")
    m4.metric("Salto en el día 5 (visitas)", f"{es(salto*100,0,'')}", delta=f"{es(salto/izq5*100,1,'')} %", delta_color="inverse")
    if abs(salto) < 1e-6:
        st.success("Con este valor las visitas son continuas en el día 5: no hay salto.")
    else:
        st.warning(f"Las visitas **no son continuas** en t = 5: por la izquierda {es(izq5*100,0,'')} y por la derecha {es(der5*100,0,'')}. "
                   f"El ritmo pasa de {es(d_izq*100,0,'+')} a {es(d_der*100,0,'+')} visitas por día cada día.")
    st.caption("Como la función salta en t = 5, **no es derivable allí**: solo se pueden dar los ritmos de cada lado.")

# ============================ TAB 3 =========================================
with tab3:
    st.markdown("**Un solo valor c que resuma la semana** (el que menos error cuadrático acumula frente a los datos, en cientos de visitas).")
    datos = DATOS_BASE + [430 + 10 * k]
    df = pd.DataFrame({"Día": list(range(1, 8)), "Visitas (cientos)": datos})
    editado = st.data_editor(df, hide_index=True, disabled=["Día"], width="content",
                             key=f"datos_{k}", num_rows="fixed")
    x = editado["Visitas (cientos)"].to_numpy(dtype=float)
    n = len(x)
    c_opt = float(x.mean())
    S = lambda c: np.sum((c - x[:, None]) ** 2, axis=0)
    cs = np.linspace(x.min() - 50, x.max() + 50, 400)
    cc = st.slider("Pruebe un valor c (cientos)", float(x.min()), float(x.max()), float(round(c_opt)), 1.0)
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Error cuadrático S(c)", "Su ritmo de cambio S′(c)"))
    fig3.add_trace(go.Scatter(x=cs, y=S(cs), line=dict(color=TEAL, width=3), name="S(c)"), 1, 1)
    fig3.add_trace(go.Scatter(x=[c_opt], y=S(np.array([c_opt])), mode="markers", name="mínimo",
                              marker=dict(color=NARANJA, size=12, symbol="diamond")), 1, 1)
    fig3.add_trace(go.Scatter(x=[cc], y=S(np.array([cc])), mode="markers", name="su valor", marker=dict(color=AZUL, size=10)), 1, 1)
    fig3.add_trace(go.Scatter(x=cs, y=2 * n * (cs - c_opt), line=dict(color=NARANJA, width=3), name="S′(c)"), 1, 2)
    fig3.add_hline(y=0, line=dict(color=GRIS, dash="dot"), row=1, col=2)
    fig3.update_xaxes(title_text="c (cientos de visitas)")
    fig3.update_yaxes(title_text="Error (cientos²)", row=1, col=1)
    fig3.update_yaxes(title_text="Cambio del error por unidad de c", row=1, col=2)
    fig3.update_layout(height=420, showlegend=False, margin=dict(t=60, b=30))
    st.plotly_chart(fig3, width="stretch")
    q1, q2, q3 = st.columns(3)
    q1.metric("c que minimiza el error (cientos)", f"{es(c_opt,2,'')}",
              help="Es el promedio de los datos. S′(c)=2·Σ(c−xᵢ)=0 → c = Σxᵢ/n; S″=2n>0, así que es un mínimo.")
    q2.metric("Ese valor en visitas", f"{es(c_opt*100,0,'')}")
    q3.metric("Error con su c", f"{es(float(S(np.array([cc]))[0]),0,'')}",
              delta=f"{es(float(S(np.array([cc]))[0] - S(np.array([c_opt]))[0]),0,'')} frente al mínimo", delta_color="inverse")
    st.markdown("**Por qué es importante en aprendizaje automático.** Entrenar un modelo es buscar el valor que "
                "minimiza una *función de pérdida*; aquí la pérdida es el error cuadrático y el mejor valor único es el promedio. "
                "Los algoritmos de optimización siguen exactamente esta idea: avanzar en la dirección en que el error baja "
                "hasta que su ritmo de cambio es cero.")
    with st.expander("Ver el descenso paso a paso (cómo lo hace un algoritmo)"):
        eta = st.slider("Tamaño del paso", 0.01, 0.14, 0.05, 0.01)
        pasos = st.slider("Número de pasos", 0, 40, 10)
        c_iter, hist = float(x.min()), []
        for i in range(pasos + 1):
            hist.append((i, c_iter, float(2 * np.sum(c_iter - x))))
            c_iter = c_iter - eta * 2 * np.sum(c_iter - x) / n
        st.dataframe(pd.DataFrame(hist, columns=["paso", "c", "S′(c)"]).round(3), hide_index=True)

# ============================ TAB 4 =========================================
with tab4:
    st.subheader("Decisión para la organización")
    peak_post = float(v_exp(5)) * 100
    plan = tope * (1 + margen / 100)
    st.markdown(f"""
**1. ¿Qué modelo usar para dimensionar los servidores?** El **modelo corregido V\\***: llega a un tope de
**{es(tope,0,'')} visitas por día** y deja un margen frente al pico de la campaña ({es(peak_post,0,'')} visitas el día 5).
Con un margen de seguridad de {margen} %, la capacidad a planear es **{es(plan,0,'')} visitas por día**.
""")
    if t_cap is not None and t_cap <= TMAX:
        inicio = max(t_cap - anticipacion, 0.0)
        st.markdown(f"""
**2. ¿Cuándo escalar?** Con una capacidad actual de {es(capacidad,0,'')} visitas diarias, la demanda la alcanza en el
**día {es(t_cap,2,'')}**. Como ampliar tarda {es(anticipacion,1,'')} día(s), hay que **empezar la ampliación a más tardar el día {es(inicio,2,'')}**.
El crecimiento es más rápido el **día {es(t_star,2,'')}** (+{es(rate_max*100,0,'')} visitas por día cada día): a partir de ahí
cada día se suman menos visitas nuevas que el día anterior, aunque el total sigue subiendo. Por eso conviene **vigilar el tráfico desde ese día**
y tener la ampliación lista antes del día {es(t_cap,2,'')}.
""")
    elif capacidad >= tope:
        st.success("La capacidad actual ya cubre el tope del modelo: no hace falta ampliar para este escenario.")
    else:
        st.warning("La capacidad actual es menor que las visitas del inicio: hay que ampliar desde ya.")
    if capacidad < plan:
        st.error(f"La capacidad actual ({es(capacidad,0,'')}) es menor que la recomendada ({es(plan,0,'')}): faltan {es(plan-capacidad,0,'')} visitas por día.")
    else:
        st.success("La capacidad actual cubre la capacidad recomendada.")
    st.markdown(f"""
**3. Condiciones y riesgos.**
- V\\* es el modelo que propone el equipo y no se ha comparado con visitas reales: hay que revisar las primeras visitas y recalcular **a** y **b** si no coinciden.
- Después del día 5 las visitas bajan; si la regla de caída es la del escenario, el modelo V no se acerca al tope: conviene **desescalar** y no pagar capacidad de más.
- El tope de {es(tope,0,'')} visitas no se alcanza exactamente dentro del mes (el día 30 está a {es((K-float(v_cor(30, K, a, b)))*100,1,'')} visitas del tope), pero describe bien la tendencia.
""")

# ============================ TAB 5 =========================================
with tab5:
    st.subheader("Dominio de los modelos (Pregunta 1)")
    st.table(pd.DataFrame({
        "Modelo": ["V(t) = 200·1,5^t", "V(t) = 900·0,8^(t−5)", "V*(t) = 2000/(1+9·0,6^t)"],
        "Dominio": ["0 ≤ t ≤ 5", "5 < t ≤ 30", "0 ≤ t ≤ 30"],
        "Qué fija cada extremo": [
            "t = 0 es el inicio de la campaña; t = 5 es el día en que termina.",
            "Empieza al terminar la campaña (t = 5) y llega a t = 30, el final del primer mes.",
            "t = 0 es el inicio; t = 30 es el mes que se analiza.",
        ]}))
    st.subheader("Derivadas (calculadas a mano y con SymPy)")
    tt_, K_, a_, b_ = sp.symbols("t K a b", positive=True)
    exp_ = 200 * sp.Rational(3, 2) ** tt_
    cor_ = K_ / (1 + a_ * b_ ** tt_)
    post_ = 900 * sp.Rational(4, 5) ** (tt_ - 5)
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"V(t)=200(1{,}5)^t \;\Rightarrow\; V'(t)=200\,\ln(1{,}5)\,(1{,}5)^t")
        st.latex(r"V(t)=900(0{,}8)^{t-5} \;\Rightarrow\; V'(t)=900\,\ln(0{,}8)\,(0{,}8)^{t-5}")
        st.latex(r"V^*(t)=\frac{K}{1+a\,b^t}\;\Rightarrow\; V^{*\prime}(t)=\frac{-K\,a\,\ln(b)\,b^t}{(1+a\,b^t)^2}")
        st.latex(r"V^{*\prime\prime}(t)=\frac{-K\,a\,(\ln b)^2\,b^t\,(1-a\,b^t)}{(1+a\,b^t)^3}")
        st.latex(r"V^{*\prime\prime}(t)=0 \iff a\,b^t=1 \iff t=\frac{\ln a}{-\ln b}\approx 4{,}30")
    with col_b:
        st.caption("SymPy deriva lo mismo (comprobación de las fórmulas a mano):")
        st.latex(sp.latex(sp.Eq(sp.Derivative(exp_, tt_), sp.diff(exp_, tt_))))
        st.latex(sp.latex(sp.Eq(sp.Derivative(post_, tt_), sp.diff(post_, tt_))))
        st.latex(sp.latex(sp.Eq(sp.Derivative(cor_, tt_), sp.diff(cor_, tt_))))
    st.subheader("Verificación numérica")
    ts_ = np.array([1.0, 3.0, 4.3, 6.0, 10.0])
    h = 1e-5
    aprox = (v_cor(ts_ + h, 2000, 9, 0.6) - v_cor(ts_ - h, 2000, 9, 0.6)) / (2 * h)
    st.dataframe(pd.DataFrame({"t": ts_, "V*′(t) con la fórmula": dv_cor(ts_, 2000, 9, 0.6),
                               "V*′(t) con diferencias (definición)": aprox,
                               "diferencia": np.abs(dv_cor(ts_, 2000, 9, 0.6) - aprox)}), hide_index=True)
    st.caption("La derivada es el límite de (V(t+h) − V(t−h)) / 2h cuando h → 0; con h muy pequeño los dos valores coinciden.")
    st.subheader("Código de las funciones")
    st.code("\n".join(inspect.getsource(f) for f in (v_exp, dv_exp, v_post, dv_post, v_cor, dv_cor, d2v_cor, t_inflexion, t_para_nivel)),
            language="python")
