"""
app.py - Interface do app "Trajetória térmica - Linha 711".

Projeto individual de Cálculo Numérico (UFMT) - José Viana Junior.
Aplica métodos numéricos (Ruggiero & Lopes) à proposta do TCC sobre a
exposição ao calor de usuários do transporte coletivo em Cuiabá.

Rodar no computador:  streamlit run app.py
"""

import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dados as dd
import metodos as mn
import normas as nm
import relatorio as rel

st.set_page_config(page_title="Trajetória térmica · Linha 711", page_icon="🚌",
                   layout="centered")

CORES = {"Espera no ponto": "#d9480f", "Interior do veículo": "#1971c2",
         "Chegada (estação)": "#2b8a3e"}


# ----------------------------------------------------------------------------
# Funções auxiliares de formatação e gráficos
# ----------------------------------------------------------------------------
def br(x, nd=2):
    """Número no formato brasileiro (vírgula decimal)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def hhmmss(h):
    s = int(round((h % 24) * 3600))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def hhmm(h):
    return hhmmss(h)[:5]


def estilo(fig, y="", x="", altura=360):
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                      xaxis_title=x, yaxis_title=y, hovermode="x unified")
    return fig


def eixo_horas(fig, xmin, xmax, passo=1):
    ticks = np.arange(math.floor(xmin), math.ceil(xmax) + 1e-9, passo)
    fig.update_xaxes(tickvals=ticks, ticktext=[hhmm(t) for t in ticks])
    return fig


# ----------------------------------------------------------------------------
# Barra lateral: dados de entrada e parâmetros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ajustes")

    st.subheader("Série horária de temperatura (INMET)")
    fonte = st.radio("Fonte", ["Exemplo sintético", "Carregar CSV"], key="fonte")
    serie = dd.serie_horaria_exemplo()
    rotulo_serie = "série sintética de exemplo"
    serie_real = False
    if fonte == "Carregar CSV":
        arq = st.file_uploader("CSV do INMET (dados históricos) ou simples (hora;temperatura)",
                               type=["csv", "txt", "CSV"])
        st.download_button("Baixar CSV-modelo", dd.csv_exemplo_bytes(), "modelo_serie.csv",
                           "text/csv")
        if arq is not None:
            try:
                lida = dd.ler_csv_temperatura(arq.getvalue())
                datas = sorted(lida["data"].unique())
                dia = st.selectbox("Dia", datas, index=len(datas) - 1)
                s = lida[lida["data"] == dia][["hora", "temperatura"]].reset_index(drop=True)
                if len(s) >= 4:
                    serie, serie_real, rotulo_serie = s, True, f"arquivo carregado ({dia})"
                else:
                    st.error("Esse dia tem menos de 4 leituras válidas.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Não consegui ler o arquivo: {e}")

    st.subheader("Viagem")
    inicio = st.time_input("Início da espera no ponto", dt.time(13, 0),
                           step=dt.timedelta(minutes=5))
    t_esp = int(st.number_input("Espera no ponto (min)", 1, 120, 15))
    t_vei = int(st.number_input("Interior do veículo (min)", 1, 180, 64))
    t_che = int(st.number_input("Chegada / estação (min)", 1, 120, 10))
    semente = int(st.number_input("Semente da simulação", 0, 9999, 0,
                                  help="Trocar a semente gera outro dia simulado."))
    arq_viagem = st.file_uploader("Medições reais da viagem (CSV) - para o TCC II", type=["csv"])

    st.subheader("Métodos")
    metodo_interp = st.radio("Interpolação de Te(t)", ["Spline cúbica natural", "Spline linear"])
    eps = st.select_slider("Tolerância ε (raízes, em horas)",
                           options=[1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10], value=1e-6,
                           format_func=lambda v: f"{v:.0e}")

    st.subheader("Parâmetros normativos (editáveis)")
    te_lim = st.number_input("NBR 15570 - Te limiar (°C)", value=nm.NBR15570_TE_LIMIAR, step=0.5)
    dt_min = st.number_input("NBR 15570 - ΔT mínimo (°C)", value=nm.NBR15570_DT_MINIMO, step=0.5)
    ref_txt = st.selectbox("Referência NR-15 (só ordem de grandeza)",
                           ["Sentado em repouso (100 W)", "Em pé em repouso (126 W)"])
    k_abr = st.number_input("Fator de abrangência k (GUM)", value=nm.K_ABRANGENCIA, step=0.1)

    with st.expander("Instrumentos (incerteza tipo B)"):
        st.caption("Valores ILUSTRATIVOS. Substituir pelas especificações do equipamento "
                   "adquirido no TCC II.")
        a_tbn = st.number_input("Exatidão tbn (°C)", value=0.5, step=0.1, min_value=0.0)
        a_tg = st.number_input("Exatidão tg (°C)", value=0.5, step=0.1, min_value=0.0)
        a_tbs = st.number_input("Exatidão tbs (°C)", value=0.5, step=0.1, min_value=0.0)
        res = st.number_input("Resolução (°C)", value=0.1, step=0.05, min_value=0.0)
        u_inmet = st.number_input("u(T_INMET) (°C)", value=0.2, step=0.05, min_value=0.0)


# ----------------------------------------------------------------------------
# Dados da viagem e da campanha (simulados ou carregados)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def viagem_simulada(serie, inicio_h, te, tv, tc, seed):
    return dd.simular_viagem(serie, inicio_h, te, tv, tc, seed=seed)


@st.cache_data(show_spinner=False)
def campanha_simulada(serie, inicio_h, te, tv, tc):
    return dd.simular_campanha(serie, 10, inicio_h, t_espera=te, t_veiculo=tv, t_chegada=tc)


inicio_h = inicio.hour + inicio.minute / 60
viagem_real = False
viagem = None
if arq_viagem is not None:
    try:
        viagem = dd.ler_csv_viagem(arq_viagem.getvalue())
        viagem_real = True
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Arquivo da viagem inválido: {e}")
if viagem is None:
    viagem = viagem_simulada(serie, inicio_h, t_esp, t_vei, t_che, semente)
resumo_camp, campanha = campanha_simulada(serie, inicio_h, t_esp, t_vei, t_che)

xs = serie["hora"].to_numpy(float)
ys = serie["temperatura"].to_numpy(float)

# ----------------------------------------------------------------------------
# 1) Interpolação: T_INMET(t) e correção local δ (Eq. 5)
# ----------------------------------------------------------------------------
spl = mn.SplineCubicaNatural(xs, ys)
if metodo_interp == "Spline cúbica natural":
    t_inmet = spl
    d_t_inmet = spl.derivada
else:
    def t_inmet(t):
        return mn.spline_linear(xs, ys, t)

    def d_t_inmet(t):
        return mn.spline_linear_derivada(xs, ys, t)

hora_v = viagem["hora"].to_numpy(float)
inmet_v = t_inmet(hora_v)
m_esp = (viagem["etapa"] == "Espera no ponto").to_numpy()
m_vei = (viagem["etapa"] == "Interior do veículo").to_numpy()
m_che = (viagem["etapa"] == "Chegada (estação)").to_numpy()
d_emb = float(np.mean(viagem["tbs"].to_numpy()[m_esp] - inmet_v[m_esp])) if m_esp.any() else np.nan
d_che = float(np.mean(viagem["tbs"].to_numpy()[m_che] - inmet_v[m_che])) if m_che.any() else np.nan
deltas = [d for d in (d_emb, d_che) if not math.isnan(d)]
delta = float(np.mean(deltas)) if deltas else 0.0
# u(δ): a diferença entre as correções é tratada como a largura total de
# uma distribuição retangular -> u = |δ_emb - δ_che| / (2√3)
u_delta = abs(d_emb - d_che) / (2 * math.sqrt(3)) if len(deltas) == 2 else 0.0


def te_func(t):
    return t_inmet(t) + delta


# ----------------------------------------------------------------------------
# 2) Raízes: instantes em que Te(t) = limiar da NBR 15570
# ----------------------------------------------------------------------------
def f_raiz(t):
    return te_func(t) - te_lim


x_min, x_max = float(xs.min()), float(xs.max())
intervalos = mn.isolar_raizes(f_raiz, x_min, x_max, n_sub=max(12, int((x_max - x_min) * 12)))
resultados_raiz = []
for (a, b) in intervalos:
    linha = {"intervalo": (a, b)}
    for nome, func in [("Bissecção", lambda: mn.bissecao(f_raiz, a, b, eps)),
                       ("Newton-Raphson", lambda: mn.newton_raphson(f_raiz, d_t_inmet, (a + b) / 2, eps)),
                       ("Secante", lambda: mn.secante(f_raiz, a, b, eps))]:
        try:
            linha[nome] = func()
        except Exception as e:  # noqa: BLE001
            linha[nome] = {"erro": str(e)}
    resultados_raiz.append(linha)
raizes = [r["Bissecção"]["raiz"] for r in resultados_raiz if "raiz" in r["Bissecção"]]

# períodos com Te acima do limiar
cortes = [x_min] + raizes + [x_max]
periodos_acima = [(cortes[i], cortes[i + 1]) for i in range(len(cortes) - 1)
                  if f_raiz((cortes[i] + cortes[i + 1]) / 2) > 0]

# Verificação ΔT >= 8 °C no interior do veículo (Eq. 4 e Eq. 9)
u_tbs = mn.incerteza_tipo_b(a_tbs, res)
u_te = math.sqrt(u_inmet ** 2 + u_delta ** 2)
u_dt = math.sqrt(u_te ** 2 + u_tbs ** 2)
U_dt = k_abr * u_dt
vei = viagem[m_vei].copy()
vei["Te estimada"] = te_func(vei["hora"].to_numpy(float))
vei["ΔT"] = vei["Te estimada"] - vei["tbs"]
vei["aplica"] = vei["Te estimada"] > te_lim
vei["situação"] = np.select(
    [~vei["aplica"], vei["ΔT"] - U_dt >= dt_min, vei["ΔT"] + U_dt < dt_min],
    ["não se aplica (Te ≤ limiar)", "atende", "não atende"], default="indeterminado (±U)")
n_aplica = int(vei["aplica"].sum())
pct_atende = 100 * (vei["situação"] == "atende").sum() / n_aplica if n_aplica else float("nan")

# ----------------------------------------------------------------------------
# 3) Integração: IBUTG médio da trajetória
# ----------------------------------------------------------------------------
y_ib = viagem["IBUTG"].to_numpy(float)
passos = np.diff(viagem["minuto"].to_numpy())
passo_ok = len(passos) > 0 and np.all(passos == passos[0])
h_min = float(passos[0]) if len(passos) else 1.0
dur_total = h_min * (len(y_ib) - 1)
I_trap = mn.trapezios_repetida(y_ib, h_min)
I_simp, simp_ajuste = mn.simpson_repetida(y_ib, h_min)
media_trap = I_trap / dur_total
media_simp = I_simp / dur_total
por_etapa = (viagem.groupby("etapa", sort=False)["IBUTG"]
             .agg(["mean", "max", "count"]).reset_index())
por_etapa.columns = ["Etapa", "IBUTG médio", "IBUTG máx", "leituras"]
por_etapa["t_j (min)"] = por_etapa["leituras"] * h_min
media_eq3 = float((por_etapa["IBUTG médio"] * por_etapa["t_j (min)"]).sum()
                  / por_etapa["t_j (min)"].sum())

m_ref = nm.M_SENTADO_REPOUSO if ref_txt.startswith("Sentado") else nm.M_EM_PE_REPOUSO
ibutg_ref = mn.spline_linear(nm.NR15_QUADRO1_M, nm.NR15_QUADRO1_IBUTG, m_ref)
excesso = np.maximum(0.0, y_ib - ibutg_ref)
dose = mn.trapezios_repetida(excesso, h_min)
min_acima = int((y_ib > ibutg_ref).sum() * h_min)
etapa_critica = por_etapa.loc[por_etapa["IBUTG médio"].idxmax()]

# ----------------------------------------------------------------------------
# 5) Incerteza (GUM) por etapa, a partir das 10 repetições
# ----------------------------------------------------------------------------
u_tbn = mn.incerteza_tipo_b(a_tbn, res)
u_tg = mn.incerteza_tipo_b(a_tg, res)
linhas_gum = []
for etapa in dd.ETAPAS:
    amostras = resumo_camp.loc[resumo_camp["etapa"] == etapa, "IBUTG médio"].to_numpy()
    if len(amostras) < 2:
        continue
    est = mn.estatistica_tipo_a(amostras)
    sol = etapa == "Espera no ponto"
    u_b = mn.incerteza_ibutg(u_tbn, u_tg, u_tbs, com_sol=sol)
    exp_ = mn.incerteza_expandida(est["u_A"], u_b, k_abr)
    linhas_gum.append({"Etapa": etapa, "n": est["n"], "média": est["media"], "s": est["s"],
                       "u_A": est["u_A"], "u_B": u_b, "u_c": exp_["u_c"], "U": exp_["U"],
                       "Resultado": f"{br(est['media'], 1)} ± {br(exp_['U'], 1)} °C"})
gum = pd.DataFrame(linhas_gum)

# ============================================================================
# INTERFACE
# ============================================================================
st.markdown("## 🚌 Trajetória térmica · Linha 711")
st.caption("Cálculo Numérico aplicado à exposição ao calor de usuários do transporte "
           "coletivo em Cuiabá-MT · José Viana Junior · UFMT")

if hora_v.min() < x_min or hora_v.max() > x_max:
    st.warning("A viagem sai do período coberto pela série horária; Te(t) fora desse "
               "período é **extrapolada** e pouco confiável.", icon="⚠️")
if not viagem_real:
    st.warning("**Dados simulados.** As abas 1 a 5 usam dados simulados até as medições do "
               "TCC II. Eles demonstram os métodos e **não são resultados de pesquisa**.",
               icon="⚠️")
st.caption("Ajustes e envio de arquivos: menu **›** no canto superior esquerdo.")

abas = st.tabs(["Início", "📝 Relatório", "1 · Interpolação", "2 · Raízes e ΔT",
                "3 · Integração", "4 · Quadrados mínimos", "5 · Incerteza", "Normas"])
(aba_ini, aba_rel, aba_1, aba_2, aba_3, aba_4, aba_5, aba_normas) = abas

# ---------------------------------------------------------------- INÍCIO
with aba_ini:
    st.subheader("Resumo")
    c1, c2 = st.columns(2)
    c1.metric("IBUTG médio da trajetória", f"{br(media_simp)} °C",
              help="Média no tempo pela Regra 1/3 de Simpson repetida (aba 3).")
    c2.metric(f"Etapa mais crítica: {etapa_critica['Etapa']}",
              f"{br(etapa_critica['IBUTG médio'])} °C", help="Maior IBUTG médio entre as etapas.")
    c3, c4 = st.columns(2)
    txt_periodo = " e ".join(f"{hhmm(a)}–{hhmm(b)}" for a, b in periodos_acima) or "nenhum"
    c3.metric(f"Te > {br(te_lim, 0)} °C", txt_periodo)
    c4.metric(f"Minutos com ΔT ≥ {br(dt_min, 0)} °C",
              "não se aplica" if math.isnan(pct_atende) else f"{br(pct_atende, 0)} %",
              help="Percentual dos minutos no veículo (com Te acima do limiar) em que "
                   "ΔT - U ≥ ΔT mínimo.")

    st.subheader("Como o app funciona")
    st.markdown(
        f"""
- **Entrada:** série horária de temperatura do INMET (*{rotulo_serie}*) e registros
  minuto a minuto da viagem (*{'arquivo carregado' if viagem_real else 'simulados'}*):
  espera no ponto → interior do veículo → chegada.
- **1 · Interpolação (cap. 5):** transforma a série horária em Te(t) contínua (Eq. 5 do TCC).
- **2 · Raízes (cap. 2):** acha os horários em que Te(t) cruza {br(te_lim, 0)} °C, quando o
  item 33.2.2 da NBR 15570 passa a valer, e verifica ΔT ≥ {br(dt_min, 0)} °C.
- **3 · Integração (cap. 7):** IBUTG médio no tempo ao longo da trajetória (Eq. 3) e
  efeito do intervalo de registro do aparelho.
- **4 · Quadrados mínimos (cap. 6):** relação entre o IBUTG interno e a temperatura dos vidros.
- **5 · Incerteza (GUM):** resultado de cada etapa na forma IBUTG ± U.
- **📝 Relatório:** você digita as medições de cada etapa e baixa um relatório técnico em PDF.
"""
    )
    st.caption("Métodos implementados do zero em Python (arquivo metodos.py), conforme "
               "Ruggiero & Lopes, *Cálculo Numérico*, 2. ed.; validados contra NumPy/SciPy.")

# ---------------------------------------------------------- RELATÓRIO
with aba_rel:
    st.subheader("📝 Avaliação de uma viagem")
    st.markdown("Digite as medições de cada etapa e baixe o **relatório técnico em PDF**. "
                "Toque numa célula para editar; use a última linha para incluir etapas "
                "(ex.: terminal intermediário).")
    exemplo = pd.DataFrame({
        "Etapa": ["Espera no ponto", "Interior do veículo", "Chegada (estação)"],
        "Duração (min)": [15.0, 64.0, 10.0],
        "Carga solar": [True, False, False],
        "tbn": [28.8, 23.0, 28.7], "tg": [48.2, 31.4, 38.7], "tbs": [36.7, 29.5, 37.5]})
    etapas_in = st.data_editor(
        exemplo, num_rows="dynamic", hide_index=True, key="editor_etapas",
        column_config={
            "Etapa": st.column_config.TextColumn("Etapa", required=True),
            "Duração (min)": st.column_config.NumberColumn("t (min)", min_value=0.0, step=1.0),
            "Carga solar": st.column_config.CheckboxColumn("Sol", default=False),
            "tbn": st.column_config.NumberColumn("tbn (°C)", format="%.1f", step=0.1),
            "tg": st.column_config.NumberColumn("tg (°C)", format="%.1f", step=0.1),
            "tbs": st.column_config.NumberColumn("tbs (°C)", format="%.1f", step=0.1),
        })
    reais = st.toggle("Estes valores são medições reais", value=False,
                      help="Desligado, o PDF sai marcado como EXEMPLO.")

    usar_dt = st.checkbox("Incluir verificação da climatização (ΔT, NBR 15570)", value=True)
    if usar_dt:
        c1, c2, c3 = st.columns(3)
        te_in = c1.number_input("Te (°C)", value=37.4, step=0.1, help="Temperatura externa "
                                "(INMET corrigida, Eq. 5).")
        ti_in = c2.number_input("Ti (°C)", value=29.5, step=0.1, help="Temperatura do ar no "
                                "interior do veículo.")
        ute_in = c3.number_input("u(Te) (°C)", value=round(u_te, 2), step=0.05, min_value=0.0)
    else:
        te_in = ti_in = None
        ute_in = 0.0

    with st.expander("Identificação (vai para o PDF)"):
        id_linha = st.text_input("Linha / sentido", "711 - Terminal Pedra 90 → Estação Alencastro")
        id_data = st.date_input("Data da medição", dt.date.today(), format="DD/MM/YYYY")
        id_faixa = st.text_input("Faixa horária", "Início da tarde (12 h - 15 h)")
        id_resp = st.text_input("Responsável", "José Viana Junior")
        id_equip = st.text_input("Equipamento (marca, modelo, nº de série)", "")
        id_calib = st.text_input("Certificado de calibração (nº e data)", "")
        id_obs = st.text_area("Condições observadas (sol, lotação, portas, climatização)", "")

    try:
        res_rel = rel.avaliar(etapas_in, u_tbn, u_tg, u_tbs, ibutg_ref, te_in, ti_in, ute_in,
                              te_lim, dt_min, k_abr)
    except Exception as e:  # noqa: BLE001
        res_rel = None
        st.error(f"Verifique os dados: {e}")

    if res_rel is not None:
        crit = res_rel["critica"]
        c1, c2 = st.columns(2)
        c1.metric("IBUTG da trajetória (Eq. 3)",
                  f"{br(res_rel['ibutg_traj'], 1)} ± {br(res_rel['U_traj'], 1)} °C")
        c2.metric(f"Etapa crítica: {crit['Etapa']}", f"{br(crit['IBUTG'], 1)} °C")
        tab_r = res_rel["tabela"][["Etapa", "IBUTG", "U", "Acima da ref."]].copy()
        tab_r["IBUTG"] = tab_r["IBUTG"].round(2)
        tab_r["U"] = tab_r["U"].round(2)
        st.dataframe(tab_r, hide_index=True)
        if "dT" in res_rel:
            d = res_rel["dT"]
            st.markdown(f"**Climatização (indicativo):** ΔT = {br(d['dT'], 1)} ± "
                        f"{br(d['U'], 1)} °C → **{d['situacao']}**")
        st.markdown("**Conclusões**")
        for c in rel.conclusoes(res_rel):
            st.markdown(f"- {c}")

        ident = {"Linha / sentido": id_linha, "Data da medição": id_data.strftime("%d/%m/%Y"),
                 "Faixa horária": id_faixa, "Responsável": id_resp,
                 "Condições observadas": id_obs}
        instrumentos = {
            "Equipamento": id_equip or "não informado",
            "Calibração": id_calib or "não informada",
            "Exatidão declarada (tbn / tg / tbs)": f"{br(a_tbn, 2)} / {br(a_tg, 2)} / "
                                                   f"{br(a_tbs, 2)} °C; resolução {br(res, 2)} °C",
            "Incerteza padrão tipo B (Eq. 6) - tbn / tg / tbs":
                f"{br(u_tbn, 3)} / {br(u_tg, 3)} / {br(u_tbs, 3)} °C",
            "Referência NR-15 (ordem de grandeza)": f"{br(ibutg_ref, 1)} °C para {int(m_ref)} W",
        }
        pdf_bytes = rel.gerar_pdf(res_rel, ident, instrumentos, dados_simulados=not reais)
        st.download_button("📄 Baixar relatório (PDF)", pdf_bytes,
                           f"relatorio_linha711_{id_data.strftime('%Y-%m-%d')}.pdf",
                           "application/pdf", type="primary")
    st.caption("Incerteza aqui é só do tipo B (uma medição por etapa). Com repetições, a aba 5 "
               "combina tipo A e tipo B. Exatidão dos instrumentos: menu › Instrumentos.")

# ---------------------------------------------------------- 1 INTERPOLAÇÃO
with aba_1:
    st.subheader("1 · Temperatura externa Te(t)")
    st.markdown("O INMET registra **de hora em hora**; no ônibus o registro é **a cada minuto**. "
                "Para estimar a temperatura externa em qualquer instante (Eq. 5) é preciso "
                "**interpolar** a série.")
    st.latex(r"T_e(t) = T_{INMET}(t) + \delta")

    mostrar_runge = st.checkbox(f"Mostrar polinômio interpolador único (grau {len(xs) - 1})")
    grade = np.linspace(x_min, x_max, 800)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="INMET (horário)",
                             marker=dict(size=8, color="#212529")))
    fig.add_trace(go.Scatter(x=grade, y=mn.spline_linear(xs, ys, grade), name="Spline linear",
                             line=dict(dash="dot", color="#868e96")))
    fig.add_trace(go.Scatter(x=grade, y=spl(grade), name="Spline cúbica natural",
                             line=dict(color="#e8590c")))
    if mostrar_runge:
        coef = mn.newton_diferencas_divididas(xs, ys)
        fig.add_trace(go.Scatter(x=grade, y=mn.avaliar_newton(coef, xs, grade),
                                 name=f"Polinômio grau {len(xs) - 1}",
                                 line=dict(color="#7048e8", width=1)))
        fig.update_yaxes(range=[ys.min() - 6, ys.max() + 6])
    eixo_horas(estilo(fig, "Temperatura (°C)", "Horário local"), x_min, x_max,
               passo=3 if x_max - x_min > 8 else 1)
    st.plotly_chart(fig)
    if mostrar_runge:
        st.info("**Fenômeno de Runge** (Ruggiero, 5.6.2): um único polinômio de grau alto "
                "passa por todos os pontos, mas oscila muito perto das extremidades. "
                "Por isso se usam **splines** (polinômios de grau baixo por trechos).")

    meias = xs[:-1] + np.diff(xs) / 2
    dif = spl(meias) - mn.spline_linear(xs, ys, meias)
    c1, c2 = st.columns(2)
    c1.metric("Maior diferença linear × cúbica", f"{br(np.max(np.abs(dif)))} °C",
              f"em {hhmm(meias[np.argmax(np.abs(dif))])}", delta_color="off")
    c2.metric("Método usado no app", metodo_interp.replace(" natural", ""))

    st.markdown("**Correção local δ** (diferença entre a rota e a estação do INMET)")
    c1, c2, c3 = st.columns(3)
    c1.metric("δ embarque", f"{br(d_emb)} °C")
    c2.metric("δ chegada", f"{br(d_che)} °C")
    c3.metric("δ adotado", f"{br(delta)} °C", f"u(δ) = {br(u_delta)} °C", delta_color="off")
    st.caption("δ = média das diferenças entre a temperatura medida (tbs) e T_INMET(t) no "
               "embarque e na chegada. u(δ): diferença entre as duas correções tratada como "
               "largura de uma distribuição retangular, u = |δ_emb − δ_che| / (2√3).")

    with st.expander("Sistema linear da spline cúbica (resolvido por Gauss, cap. 3)"):
        st.latex(r"h_{i-1}g_{i-1} + 2(h_{i-1}+h_i)\,g_i + h_i g_{i+1} = "
                 r"6\left[\frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}}\right]")
        st.write(f"Sistema {spl.A.shape[0]} × {spl.A.shape[0]} (tridiagonal), com "
                 f"g₀ = gₙ = 0 (spline natural). Resíduo ‖A·g − b‖ = "
                 f"{np.linalg.norm(spl.A @ spl.g[1:-1] - spl.b):.1e}")
        st.dataframe(pd.DataFrame(spl.A).round(2), hide_index=True)

    with st.expander("Validação contra bibliotecas"):
        try:
            from scipy.interpolate import CubicSpline
            ref = CubicSpline(xs, ys, bc_type="natural")(grade)
            st.write(f"Spline cúbica: maior diferença para o SciPy = "
                     f"{np.max(np.abs(ref - spl(grade))):.1e} °C")
        except ImportError:
            st.write("SciPy não instalado.")
        st.write(f"Spline linear: maior diferença para o numpy.interp = "
                 f"{np.max(np.abs(np.interp(grade, xs, ys) - mn.spline_linear(xs, ys, grade))):.1e} °C")

# ---------------------------------------------------------- 2 RAÍZES
with aba_2:
    st.subheader(f"2 · Quando Te(t) passa de {br(te_lim, 0)} °C?")
    st.markdown(f"O item 33.2.2 da NBR 15570 só vale com Te > {br(te_lim, 0)} °C. "
                f"Os instantes de cruzamento são as **raízes** de")
    st.latex(rf"f(t) = T_e(t) - {te_lim:g} = 0")

    st.markdown("**Fase I – isolamento** (troca de sinal em subintervalos de 5 min):")
    if not intervalos:
        st.info("Te(t) não cruza o limiar no período da série.")
    else:
        st.write(", ".join(f"[{hhmm(a)}, {hhmm(b)}]" for a, b in intervalos))

        st.markdown("**Fase II – refinamento** (ε = "
                    f"{eps:.0e} h ≈ {br(eps * 3600, 4)} s):")
        tab = []
        for r in resultados_raiz:
            for nome in ["Bissecção", "Newton-Raphson", "Secante"]:
                res_m = r[nome]
                if "erro" in res_m:
                    tab.append({"Intervalo": f"{hhmm(r['intervalo'][0])}–{hhmm(r['intervalo'][1])}",
                                "Método": nome, "Raiz": "—", "Iterações": "—",
                                "Situação": res_m["erro"]})
                else:
                    tab.append({"Intervalo": f"{hhmm(r['intervalo'][0])}–{hhmm(r['intervalo'][1])}",
                                "Método": nome, "Raiz": hhmmss(res_m["raiz"]),
                                "Iterações": res_m["iteracoes"],
                                "Situação": "convergiu" if res_m["convergiu"] else "não convergiu"})
        st.dataframe(pd.DataFrame(tab), hide_index=True)
        st.caption("Bissecção converge sempre (intervalo com troca de sinal), mas devagar. "
                   "Newton usa a derivada S'(t) da spline e converge muito mais rápido. "
                   "A Secante dispensa a derivada.")

        for i, r in enumerate(resultados_raiz):
            with st.expander(f"Iterações – raiz {i + 1} ({hhmm(r['intervalo'][0])}–"
                             f"{hhmm(r['intervalo'][1])})"):
                for nome in ["Bissecção", "Newton-Raphson", "Secante"]:
                    if "tabela" in r[nome]:
                        st.markdown(f"*{nome}*")
                        st.dataframe(pd.DataFrame(r[nome]["tabela"]), hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grade, y=te_func(grade) - te_lim, name="f(t)",
                             line=dict(color="#e8590c")))
    fig.add_hline(y=0, line_color="#868e96")
    if raizes:
        fig.add_trace(go.Scatter(x=raizes, y=[0] * len(raizes), mode="markers", name="Raízes",
                                 marker=dict(size=11, color="#c92a2a", symbol="x")))
    eixo_horas(estilo(fig, "Te(t) − limiar (°C)", "Horário local"), x_min, x_max,
               passo=3 if x_max - x_min > 8 else 1)
    st.plotly_chart(fig)

    st.subheader(f"Verificação indicativa ΔT ≥ {br(dt_min, 0)} °C (NBR 15570, 33.2.2)")
    st.latex(r"\Delta T = T_e - T_i \geq \Delta T_{min}")
    st.latex(r"u(\Delta T)=\sqrt{u^2(T_e)+u^2(T_i)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("u(Te)", f"{br(u_te)} °C")
    c2.metric("u(Ti)", f"{br(u_tbs)} °C")
    c3.metric(f"U(ΔT), k={br(k_abr, 1)}", f"{br(U_dt)} °C")
    fig = go.Figure()
    cores_sit = {"atende": "#2b8a3e", "não atende": "#c92a2a",
                 "indeterminado (±U)": "#f08c00", "não se aplica (Te ≤ limiar)": "#adb5bd"}
    for sit, g in vei.groupby("situação"):
        fig.add_trace(go.Scatter(x=g["hora"], y=g["ΔT"], mode="markers", name=sit,
                                 marker=dict(color=cores_sit.get(sit, "#495057"), size=7),
                                 error_y=dict(type="constant", value=U_dt, thickness=0.6)))
    fig.add_hline(y=dt_min, line_dash="dash", line_color="#212529",
                  annotation_text=f"ΔT mínimo = {br(dt_min, 0)} °C")
    if len(vei):
        eixo_horas(estilo(fig, "ΔT (°C)", "Horário"), vei["hora"].min(), vei["hora"].max(),
                   passo=0.25)
    st.plotly_chart(fig)
    st.dataframe(vei["situação"].value_counts().rename_axis("Situação")
                 .reset_index(name="minutos"), hide_index=True)
    st.caption("Comparação **indicativa**: Te é estimada (INMET + δ), não medida ao lado do "
               "veículo. Não é ensaio de conformidade da climatização.")

# ---------------------------------------------------------- 3 INTEGRAÇÃO
with aba_3:
    st.subheader("3 · IBUTG ao longo da trajetória")
    st.latex(r"\small IBUTG_{sol} = 0{,}7\,t_{bn} + 0{,}2\,t_g + 0{,}1\,t_{bs}")
    st.latex(r"\small IBUTG_{sombra} = 0{,}7\,t_{bn} + 0{,}3\,t_g")
    fig = go.Figure()
    for etapa, g in viagem.groupby("etapa", sort=False):
        fig.add_trace(go.Scatter(x=g["minuto"], y=g["IBUTG"], mode="lines+markers",
                                 name=etapa, marker=dict(size=4),
                                 line=dict(color=CORES.get(etapa, "#495057"))))
    fig.add_hline(y=ibutg_ref, line_dash="dash", line_color="#c92a2a",
                  annotation_text=f"NR-15, {int(m_ref)} W: {br(ibutg_ref)} °C (referência)")
    st.plotly_chart(estilo(fig, "IBUTG (°C)", "Minutos desde o início da espera"))

    st.markdown("**IBUTG médio no tempo** (Eq. 3 do TCC na forma contínua):")
    st.latex(r"\overline{IBUTG} = \frac{1}{T}\int_0^T IBUTG(t)\,dt")
    comp = pd.DataFrame([
        {"Método": "Eq. 3 (média de cada etapa × t_j)", "IBUTG médio (°C)": media_eq3},
        {"Método": "Trapézios repetida (cap. 7.2.2)", "IBUTG médio (°C)": media_trap},
        {"Método": "1/3 de Simpson repetida (cap. 7.2.4)", "IBUTG médio (°C)": media_simp},
    ])
    comp["IBUTG médio (°C)"] = comp["IBUTG médio (°C)"].round(3)
    st.dataframe(comp, hide_index=True)
    if simp_ajuste:
        st.caption("Número de subintervalos ímpar: Simpson nos m−1 primeiros + Trapézio "
                   "no último.")
    if not passo_ok:
        st.error("Os registros não têm intervalo constante; a integração assume passo fixo.")

    st.dataframe(por_etapa.round(2), hide_index=True)

    st.markdown(f"**Exposição acima da referência ({br(ibutg_ref)} °C)**")
    c1, c2 = st.columns(2)
    c1.metric("Tempo acima", f"{min_acima} min")
    c2.metric("Excesso acumulado", f"{br(dose, 1)} °C·min",
              help="∫ max(0, IBUTG − referência) dt, pela Regra dos Trapézios.")
    st.caption("A referência da NR-15 é ocupacional e serve só como ordem de grandeza.")

    st.markdown("**Qual intervalo de registro o aparelho precisa ter?**")
    st.caption("Reamostra a série a cada h minutos (como um registrador mais lento) e "
               "compara com o registro a cada 1 min no mesmo trecho de tempo.")
    estudo = []
    for passo in [1, 2, 3, 5, 10, 15]:
        amostra = y_ib[::passo]
        if len(amostra) < 3:
            continue
        n_cob = passo * (len(amostra) - 1)          # pontos de 1 min cobertos
        T = n_cob * h_min
        ref_trecho = mn.trapezios_repetida(y_ib[:n_cob + 1], h_min) / T
        mt = mn.trapezios_repetida(amostra, passo * h_min) / T
        ms = mn.simpson_repetida(amostra, passo * h_min)[0] / T
        estudo.append({"h (min)": passo * h_min, "pontos": len(amostra),
                       "trecho (min)": T, "Trapézios (°C)": mt, "Simpson (°C)": ms,
                       "|erro| Trap. (°C)": abs(mt - ref_trecho),
                       "|erro| Simp. (°C)": abs(ms - ref_trecho)})
    estudo = pd.DataFrame(estudo)
    st.dataframe(estudo.round(3), hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=estudo["h (min)"], y=estudo["|erro| Trap. (°C)"],
                             mode="lines+markers", name="Trapézios"))
    fig.add_trace(go.Scatter(x=estudo["h (min)"], y=estudo["|erro| Simp. (°C)"],
                             mode="lines+markers", name="Simpson"))
    st.plotly_chart(estilo(fig, "|erro| vs. h = 1 min (°C)", "Intervalo de registro h (min)", 300))
    st.caption("Com dados medidos (com ruído), o erro mistura truncamento do método e perda de "
               "informação pela amostragem: picos curtos (abertura de portas) somem com h grande. "
               "Simpson tem erro de ordem h⁴ só para funções suaves; com ruído e picos ele não "
               "supera o Trapézio, e pode ser pior.")

# ---------------------------------------------------------- 4 QUADRADOS MÍNIMOS
with aba_4:
    st.subheader("4 · IBUTG interno × temperatura dos vidros")
    st.markdown("O TCC prevê regressão só onde há fundamento físico, como a relação entre o "
                "IBUTG interno e a temperatura dos vidros (radiação para o interior).")
    grau = st.radio("Grau do polinômio", [1, 2], horizontal=True)
    base = campanha[campanha["etapa"] == "Interior do veículo"].dropna(subset=["ts_vidro"])
    x_aj = base["ts_vidro"].to_numpy(float)
    y_aj = base["IBUTG"].to_numpy(float)
    aj = mn.minimos_quadrados_polinomio(x_aj, y_aj, grau)
    termos = " + ".join([f"{br(aj['coef'][0], 3)}"] +
                        [f"{br(c, 4)}·tₛ{'²' if i == 2 else ''}" for i, c in
                         enumerate(aj["coef"][1:], start=1)])
    st.markdown(f"**IBUTG ≈ {termos}**")
    c1, c2 = st.columns(2)
    c1.metric("R²", br(aj["r2"], 3))
    c2.metric("Pontos (10 dias)", len(x_aj))
    xx = np.linspace(x_aj.min(), x_aj.max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_aj, y=y_aj, mode="markers", name="Leituras (1/min)",
                             marker=dict(size=4, color="#1971c2", opacity=0.5)))
    fig.add_trace(go.Scatter(x=xx, y=mn.avaliar_polinomio(aj["coef"], xx), name="Ajuste",
                             line=dict(color="#c92a2a", width=3)))
    st.plotly_chart(estilo(fig, "IBUTG interno (°C)", "Temperatura dos vidros tₛ (°C)"))
    if grau == 1:
        st.caption(f"Leitura: cada +1 °C nos vidros corresponde a cerca de "
                   f"{br(aj['coef'][1], 2)} °C no IBUTG interno (dados simulados).")
    with st.expander("Equações normais (resolvidas por Gauss, cap. 3)"):
        st.write("Matriz A:")
        st.dataframe(pd.DataFrame(aj["A"]).round(1), hide_index=True)
        st.write("Vetor b:", np.round(aj["b"], 1).tolist())
        ref = np.polyfit(x_aj, y_aj, grau)[::-1]
        st.write(f"Validação: maior diferença para numpy.polyfit = "
                 f"{np.max(np.abs(ref - aj['coef'])):.1e}")
    fig = go.Figure(go.Scatter(x=x_aj, y=aj["residuos"], mode="markers",
                               marker=dict(size=4, color="#495057")))
    fig.add_hline(y=0, line_color="#c92a2a")
    st.plotly_chart(estilo(fig, "Resíduo (°C)", "tₛ (°C)", 250))
    st.caption("Resíduos sem padrão (espalhados em torno de zero) indicam que o grau escolhido "
               "é adequado (Ruggiero, 6.3.1 – testes de alinhamento).")

# ---------------------------------------------------------- 5 INCERTEZA
with aba_5:
    st.subheader("5 · Resultado por etapa: IBUTG ± U")
    st.markdown("10 repetições simuladas por etapa (Seção 3.3 do TCC: 10 dias úteis).")
    st.latex(r"u_B=\sqrt{\left(\tfrac{a}{\sqrt3}\right)^2+\left(\tfrac{r}{2\sqrt3}\right)^2}"
             r"\qquad u_A=\tfrac{s}{\sqrt n}")
    st.latex(r"u_c=\sqrt{u_A^2+u_B^2}\qquad U=k\,u_c")
    st.latex(r"\small u^2_{IBUTG}=(0{,}7\,u_{bn})^2+(0{,}2\,u_{g})^2+(0{,}1\,u_{bs})^2")
    if len(gum):
        st.dataframe(gum[["Etapa", "Resultado"]], hide_index=True)
        with st.expander("Componentes da incerteza"):
            st.dataframe(gum.drop(columns=["Resultado"]).round(3), hide_index=True)
        fig = go.Figure(go.Bar(x=gum["Etapa"], y=gum["média"],
                               marker_color=[CORES.get(e, "#495057") for e in gum["Etapa"]],
                               error_y=dict(type="data", array=gum["U"], thickness=2, width=10)))
        fig.add_hline(y=ibutg_ref, line_dash="dash", line_color="#c92a2a",
                      annotation_text=f"NR-15 {int(m_ref)} W (ref.)")
        fig.update_yaxes(range=[float((gum["média"] - gum["U"]).min()) - 3,
                                max(float((gum["média"] + gum["U"]).max()), ibutg_ref) + 1.5])
        st.plotly_chart(estilo(fig, "IBUTG (°C)", "", 340))
    st.caption("Exatidão e resolução são ilustrativas (menu › Instrumentos). No TCC II entram "
               "as especificações e o certificado de calibração do equipamento.")

# ---------------------------------------------------------- NORMAS
with aba_normas:
    st.subheader("Normas e referências usadas")
    st.dataframe(pd.DataFrame(nm.REFERENCIAS, columns=["Documento", "Uso no app"]),
                 hide_index=True)
    st.markdown(f"**NR-15, Anexo 3 – Quadro 1 (trecho)**. Para M = {int(m_ref)} W, o valor por "
                f"**interpolação linear** (spline linear, cap. 5) é **{br(ibutg_ref)} °C**.")
    st.dataframe(pd.DataFrame({"M (W)": nm.NR15_QUADRO1_M.astype(int),
                               "IBUTG máx (°C)": nm.NR15_QUADRO1_IBUTG}), hide_index=True)
    st.caption("Conferir na norma o critério para taxas metabólicas que não constam do quadro. "
               "O Anexo 3 está em revisão pelo MTE (consulta pública de 2024).")
    st.caption("Só os parâmetros numéricos são usados aqui, com a fonte indicada; o texto das "
               "normas ABNT/ISO é protegido e não é reproduzido. Edições conferidas em nov/2026.")
    st.download_button("Baixar modelo de CSV da viagem (TCC II)", dd.viagem_para_csv(viagem),
                       "modelo_viagem.csv", "text/csv")
