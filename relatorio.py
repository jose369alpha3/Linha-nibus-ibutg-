"""
relatorio.py - Avaliação de uma viagem a partir de valores digitados
(leituras de IBUTG ou temperaturas tbn, tg, tbs) e relatório técnico em PDF.

Para cada etapa: n leituras -> média, desvio padrão, u_A, u_B, u_c e U
(Eqs. 6 a 14 do TCC). Trajetória: média ponderada pelo tempo (Eq. 3).

O PDF NÃO é laudo de insalubridade (NR-15) nem ensaio de conformidade
(NBR 15570). É um relatório acadêmico de avaliação da carga térmica.
"""

import datetime as dt
import math
import re

import numpy as np
import pandas as pd
from fpdf import FPDF

import metodos as mn
import normas as nm


# ----------------------------------------------------------------------------
# Leitura do que o usuário digitou
# ----------------------------------------------------------------------------
def parse_leituras(txt):
    """'33,1; 33,4 33,0' -> [33.1, 33.4, 33.0].
    Separador: ponto e vírgula, espaço ou quebra de linha. Decimal: vírgula ou ponto."""
    if txt is None or (isinstance(txt, float) and math.isnan(txt)):
        return []
    vals = []
    for p in re.split(r"[;\s]+", str(txt).strip()):
        p = p.strip().rstrip(",")
        if not p:
            continue
        if p.count(",") > 1 or (p.count(",") == 1 and p.count(".") == 1):
            raise ValueError(f"não entendi \"{p}\". Separe as leituras com espaço ou ; "
                             f"e use vírgula só como decimal (ex.: 33,1; 33,4).")
        try:
            vals.append(float(p.replace(",", ".")))
        except ValueError:
            raise ValueError(f"\"{p}\" não é um número.") from None
    return vals


def leituras_de_ibutg(tab):
    """Tabela com colunas Etapa, Duração (min), Carga solar, Leituras -> formato longo."""
    linhas = []
    for _, r in tab.iterrows():
        if pd.isna(r.get("Etapa")) or not str(r["Etapa"]).strip():
            continue
        try:
            vals = parse_leituras(r.get("Leituras"))
        except ValueError as e:
            raise ValueError(f"Etapa \"{r['Etapa']}\": {e}") from None
        if not vals:
            continue
        for v in vals:
            linhas.append({"Etapa": str(r["Etapa"]).strip(), "t": r.get("Duração (min)"),
                           "Sol": bool(r.get("Carga solar")) if pd.notna(r.get("Carga solar")) else False,
                           "IBUTG": v})
    return pd.DataFrame(linhas)


def leituras_de_temperaturas(tab):
    """Tabela com Etapa, Duração (min), Carga solar, tbn, tg, tbs. Linhas com o mesmo
    nome de etapa são repetições; duração e carga solar valem pela 1ª linha da etapa."""
    tab = tab.dropna(subset=["Etapa", "tbn", "tg"]).copy()
    tab["Etapa"] = tab["Etapa"].astype(str).str.strip()
    tab = tab[tab["Etapa"] != ""]
    if tab.empty:
        return pd.DataFrame()
    primeira = tab.groupby("Etapa", sort=False).first()
    linhas = []
    for _, r in tab.iterrows():
        sol = bool(primeira.loc[r["Etapa"], "Carga solar"]) \
            if pd.notna(primeira.loc[r["Etapa"], "Carga solar"]) else False
        if sol and pd.isna(r["tbs"]):
            raise ValueError(f"Etapa \"{r['Etapa']}\" tem carga solar e precisa de tbs (Eq. 1).")
        ib = float(nm.ibutg(r["tbn"], r["tg"], r["tbs"], True)) if sol \
            else float(nm.ibutg(r["tbn"], r["tg"], com_sol=False))
        linhas.append({"Etapa": r["Etapa"], "t": primeira.loc[r["Etapa"], "Duração (min)"],
                       "Sol": sol, "IBUTG": ib})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Cálculos (Eqs. 3, 4, 6-14 do TCC)
# ----------------------------------------------------------------------------
def avaliar(leituras, u_b_func, ibutg_ref, te=None, ti=None, u_te=0.0, u_ti=0.0,
            te_lim=32.0, dt_min=8.0, k=2.0):
    """leituras: DataFrame longo (Etapa, t, Sol, IBUTG) - uma linha por leitura.
    u_b_func(sol) -> incerteza padrão tipo B do IBUTG de uma leitura."""
    if leituras is None or leituras.empty:
        raise ValueError("Preencha pelo menos uma etapa com leituras.")
    linhas = []
    for etapa, g in leituras.groupby("Etapa", sort=False):
        t = g["t"].iloc[0]
        if pd.isna(t) or float(t) <= 0:
            raise ValueError(f"Informe a duração (min) da etapa \"{etapa}\".")
        sol = bool(g["Sol"].iloc[0])
        est = mn.estatistica_tipo_a(g["IBUTG"].to_numpy(float))
        u_a = est["u_A"] if est["n"] > 1 else 0.0
        u_b = u_b_func(sol)
        exp_ = mn.incerteza_expandida(u_a, u_b, k)
        linhas.append({"Etapa": etapa, "t (min)": float(t), "Sol": sol, "n": est["n"],
                       "média": est["media"], "s": est["s"], "u_A": u_a, "u_B": u_b,
                       "u_c": exp_["u_c"], "U": exp_["U"],
                       "mín": float(g["IBUTG"].min()), "máx": float(g["IBUTG"].max())})
    tab = pd.DataFrame(linhas)
    tab["Acima da ref."] = tab["média"] > ibutg_ref

    T = float(tab["t (min)"].sum())
    w = tab["t (min)"] / T
    media_traj = float((w * tab["média"]).sum())                     # Eq. 3
    # Tipo A: etapas independentes -> soma quadrática.
    # Tipo B: mesmo instrumento -> erro sistemático correlacionado -> soma linear.
    u_a_traj = float(np.sqrt(((w * tab["u_A"]) ** 2).sum()))
    u_b_traj = float((w * tab["u_B"]).sum())
    u_c_traj = math.sqrt(u_a_traj ** 2 + u_b_traj ** 2)

    out = {"tabela": tab, "leituras": leituras, "T": T, "ibutg_traj": media_traj,
           "u_traj": u_c_traj, "U_traj": k * u_c_traj, "critica": tab.loc[tab["média"].idxmax()],
           "ibutg_ref": ibutg_ref, "k": k}

    if te is not None and ti is not None:
        d = te - ti
        U_d = k * math.sqrt(u_te ** 2 + u_ti ** 2)                   # Eq. 9
        if te <= te_lim:
            sit = f"não se aplica (Te = {te:.1f} °C ≤ {te_lim:.0f} °C)"
        elif d - U_d >= dt_min:
            sit = "atende"
        elif d + U_d < dt_min:
            sit = "não atende"
        else:
            sit = "indeterminado (dentro da incerteza)"
        out["dT"] = {"Te": te, "Ti": ti, "dT": d, "U": U_d, "situacao": sit,
                     "te_lim": te_lim, "dt_min": dt_min}
    return out


def conclusoes(res):
    c = []
    crit = res["critica"]
    c.append(f"A etapa de maior carga térmica foi \"{crit['Etapa']}\", com IBUTG de "
             f"{crit['média']:.1f} ± {crit['U']:.1f} °C (n = {int(crit['n'])}).")
    c.append(f"O IBUTG médio ponderado no tempo da trajetória (Eq. 3), em {res['T']:.0f} min, "
             f"foi de {res['ibutg_traj']:.1f} ± {res['U_traj']:.1f} °C (k = {res['k']:g}).")
    acima = res["tabela"][res["tabela"]["Acima da ref."]]
    if len(acima):
        c.append(f"Etapa(s) com média acima da referência de {res['ibutg_ref']:.1f} °C (NR-15, "
                 f"usada só como ordem de grandeza): " + ", ".join(acima["Etapa"]) + ".")
    else:
        c.append(f"Nenhuma etapa teve média acima da referência de {res['ibutg_ref']:.1f} °C "
                 f"(NR-15, usada só como ordem de grandeza).")
    um = res["tabela"][res["tabela"]["n"] < 2]
    if len(um):
        c.append("Etapa(s) com uma única leitura (sem incerteza tipo A): "
                 + ", ".join(um["Etapa"]) + ".")
    if "dT" in res:
        d = res["dT"]
        c.append(f"Climatização: ΔT = Te − Ti = {d['dT']:.1f} ± {d['U']:.1f} °C; em relação ao "
                 f"mínimo de {d['dt_min']:.0f} °C da NBR 15570 (item 33.2.2), a situação é: "
                 f"{d['situacao']} (comparação indicativa).")
    return c


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
_TROCAS = {"→": "->", "←": "<-", "Δ": "Delta ", "≥": ">=", "≤": "<=", "−": "-", "–": "-",
           "—": "-", "·": ".", "×": "x", "√": "raiz", "²": "2", "“": '"', "”": '"', "’": "'",
           "…": "..."}


def _t(s):
    """Converte para Latin-1 (fontes padrão do PDF)."""
    s = str(s)
    for a, b in _TROCAS.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _n(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")


class _PDF(FPDF):
    gerado_em = ""

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120)
        self.cell(0, 5, _t(f"Relatório acadêmico gerado pelo app Trajetória térmica - Linha 711 "
                           f"em {self.gerado_em} | página {self.page_no()}"), align="C")


def gerar_pdf(res, ident, instrumentos, dados_simulados=False, modo=""):
    pdf = _PDF(format="A4")
    pdf.gerado_em = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def titulo(txt):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(230, 238, 247)
        pdf.cell(W, 7, _t(txt), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def par(txt, tam=9.5, estilo=""):
        pdf.set_font("Helvetica", estilo, tam)
        pdf.multi_cell(W, 4.6, _t(txt), new_x="LMARGIN", new_y="NEXT")

    def tabela(cab, linhas, larg):
        pdf.set_font("Helvetica", "B", 8)
        for c, l in zip(cab, larg):
            pdf.cell(l, 6, _t(c), border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for ln in linhas:
            for v, l in zip(ln, larg):
                pdf.cell(l, 6, _t(v), border=1, align="C")
            pdf.ln()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(W, 7, _t("RELATÓRIO TÉCNICO DE AVALIAÇÃO DA CARGA TÉRMICA NA TRAJETÓRIA "
                            "DO USUÁRIO DO TRANSPORTE COLETIVO"), align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    if dados_simulados:
        pdf.set_text_color(180, 0, 0)
        par("ATENÇÃO: valores de exemplo - não são medições reais.", 9.5, "B")
        pdf.set_text_color(0)

    titulo("1. Identificação")
    for k_, v in ident.items():
        par(f"{k_}: {v if str(v).strip() else '-'}")

    titulo("2. Base normativa e método")
    par("IBUTG com carga solar: 0,7 tbn + 0,2 tg + 0,1 tbs; sem carga solar: 0,7 tbn + 0,3 tg "
        "(NHO 06, Fundacentro, 3. ed., 2025; ISO 7243:2017). Medição conforme ISO 7726:2025. "
        "Por etapa: média das leituras, desvio padrão amostral e incerteza tipo A (s/raiz n), "
        "combinada com a tipo B do instrumento e expandida com fator k (JCGM 100:2008 - GUM). "
        "Trajetória: média ponderada pelo tempo de permanência em cada etapa. Climatização: ABNT "
        "NBR 15570:2021, item 33.2.2 (comparação indicativa). Referência ocupacional: NR-15, "
        "Anexo 3, Quadro 1 (Portaria SEPRT 1.359/2019), apenas como ordem de grandeza.")
    if modo:
        par(f"Forma de entrada dos dados: {modo}.")

    titulo("3. Instrumentação e incerteza tipo B")
    for k_, v in instrumentos.items():
        par(f"{k_}: {v}")

    titulo("4. Resultados por etapa")
    t = res["tabela"]
    linhas = [[r["Etapa"][:24], _n(r["t (min)"], 0), "sim" if r["Sol"] else "não", str(int(r["n"])),
               _n(r["média"], 2), _n(r["s"], 2), _n(r["u_A"], 2), _n(r["u_B"], 2),
               f"{_n(r['média'])} ± {_n(r['U'])}"] for _, r in t.iterrows()]
    tabela(["Etapa", "t (min)", "Sol", "n", "Média", "s", "u_A", "u_B", "IBUTG ± U (°C)"],
           linhas, [44, 14, 10, 9, 17, 15, 15, 15, 41])
    pdf.ln(2)
    par(f"IBUTG da trajetória (Eq. 3): {_n(res['ibutg_traj'])} ± {_n(res['U_traj'])} °C "
        f"(k = {res['k']:g}; duração total {res['T']:.0f} min).", 10, "B")
    par("Na trajetória, a incerteza tipo A das etapas é somada em quadratura (etapas "
        "independentes) e a tipo B é somada linearmente, porque o mesmo instrumento é usado em "
        "todas (erro sistemático correlacionado) - estimativa conservadora.", 8.5, "I")

    n_sec = 5
    if "dT" in res:
        d = res["dT"]
        titulo(f"{n_sec}. Climatização do veículo (NBR 15570, item 33.2.2 - indicativo)")
        par(f"Te = {_n(d['Te'])} °C; Ti = {_n(d['Ti'])} °C; Delta T = {_n(d['dT'])} ± "
            f"{_n(d['U'])} °C; mínimo exigido = {d['dt_min']:.0f} °C quando Te > "
            f"{d['te_lim']:.0f} °C.")
        par(f"Situação: {d['situacao']}.", 10, "B")
        n_sec += 1

    titulo(f"{n_sec}. Conclusões")
    for c in conclusoes(res):
        par("- " + c)
    n_sec += 1

    titulo(f"{n_sec}. Leituras registradas (°C)")
    for etapa, g in res["leituras"].groupby("Etapa", sort=False):
        par(f"{etapa} (n = {len(g)}): " + "; ".join(_n(v, 2) for v in g["IBUTG"]), 8.5)
    n_sec += 1

    # Limitações + assinatura ficam juntas (evita página só com a assinatura)
    if pdf.get_y() + 55 > pdf.h - 15:
        pdf.add_page()
    titulo(f"{n_sec}. Limitações")
    par("Este relatório é um documento acadêmico de avaliação da carga térmica ambiental. "
        "Não é laudo de insalubridade (NR-15), que exige profissional legalmente habilitado, "
        "nem ensaio de conformidade do sistema de climatização (NBR 15570). Os limites "
        "ocupacionais da NR-15 não se aplicam ao passageiro e aparecem apenas como ordem de "
        "grandeza. A sensação térmica dos usuários não foi avaliada. A temperatura externa, "
        "quando estimada pela estação do INMET, torna a verificação da climatização indicativa.",
        8.5)

    pdf.ln(10)
    pdf.cell(W, 5, "_" * 45, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(W, 5, _t(ident.get("Responsável", "")), align="C")
    return bytes(pdf.output())
