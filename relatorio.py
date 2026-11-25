"""
relatorio.py - Cálculos da avaliação de uma viagem a partir de valores
digitados e geração do relatório técnico em PDF.

O PDF NÃO é laudo de insalubridade (NR-15) nem ensaio de conformidade
(NBR 15570). É um relatório acadêmico de avaliação da carga térmica.
"""

import datetime as dt
import math

import numpy as np
import pandas as pd
from fpdf import FPDF

import metodos as mn
import normas as nm


# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
def avaliar(etapas, u_tbn, u_tg, u_tbs, ibutg_ref, te=None, ti=None, u_te=0.0,
            te_lim=32.0, dt_min=8.0, k=2.0):
    """etapas: DataFrame com colunas Etapa, Duração (min), Carga solar, tbn, tg, tbs.

    Retorna dicionário com a tabela por etapa, o IBUTG da trajetória (Eq. 3)
    e a verificação de ΔT (Eq. 4)."""
    df = etapas.copy()
    df = df.dropna(subset=["Etapa", "Duração (min)", "tbn", "tg"])
    df = df[df["Duração (min)"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("Preencha pelo menos uma etapa com duração, tbn e tg.")
    sol = df["Carga solar"].fillna(False).astype(bool)
    if (sol & df["tbs"].isna()).any():
        raise ValueError("Etapas com carga solar precisam de tbs (Eq. 1).")

    df["IBUTG"] = np.where(sol, nm.ibutg(df["tbn"], df["tg"], df["tbs"].fillna(0), True),
                           nm.ibutg(df["tbn"], df["tg"], com_sol=False))
    df["u (tipo B)"] = [mn.incerteza_ibutg(u_tbn, u_tg, u_tbs, bool(s)) for s in sol]
    df["U"] = k * df["u (tipo B)"]
    df["Acima da ref."] = df["IBUTG"] > ibutg_ref

    T = float(df["Duração (min)"].sum())
    w = df["Duração (min)"] / T
    ib_traj = float((w * df["IBUTG"]).sum())                         # Eq. 3
    # Mesmo instrumento em todas as etapas: o erro de exatidão é sistemático
    # (correlação total), então as incertezas somam linearmente (conservador).
    u_traj = float((w * df["u (tipo B)"]).sum())

    out = {"tabela": df, "T": T, "ibutg_traj": ib_traj, "u_traj": u_traj,
           "U_traj": k * u_traj, "critica": df.loc[df["IBUTG"].idxmax()],
           "ibutg_ref": ibutg_ref, "k": k}

    if te is not None and ti is not None:
        d = te - ti
        u_d = math.sqrt(u_te ** 2 + u_tbs ** 2)                     # Eq. 9
        U_d = k * u_d
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
             f"{crit['IBUTG']:.1f} ± {crit['U']:.1f} °C.")
    c.append(f"O IBUTG médio ponderado no tempo da trajetória (Eq. 3), em {res['T']:.0f} min, "
             f"foi de {res['ibutg_traj']:.1f} ± {res['U_traj']:.1f} °C (k = {res['k']:.0f}).")
    acima = res["tabela"][res["tabela"]["Acima da ref."]]
    if len(acima):
        c.append(f"Etapa(s) acima da referência de {res['ibutg_ref']:.1f} °C (NR-15, usada só "
                 f"como ordem de grandeza): " + ", ".join(acima["Etapa"]) + ".")
    else:
        c.append(f"Nenhuma etapa superou a referência de {res['ibutg_ref']:.1f} °C (NR-15, usada "
                 f"só como ordem de grandeza).")
    if "dT" in res:
        d = res["dT"]
        c.append(f"Climatização: ΔT = Te − Ti = {d['dT']:.1f} ± {d['U']:.1f} °C; em relação ao "
                 f"mínimo de {d['dt_min']:.0f} °C da NBR 15570 (item 33.2.2), a situação é: "
                 f"{d['situacao']} (comparação indicativa).")
    return c


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
_TROCAS = {"→": "->", "←": "<-", "Δ": "Delta ", "≥": ">=", "≤": "<=", "−": "-", "–": "-", "—": "-", "·": ".",
           "×": "x", "√": "raiz", "²": "2", "“": '"', "”": '"', "’": "'", "…": "..."}


def _t(s):
    """Converte para Latin-1 (fontes padrão do PDF)."""
    s = str(s)
    for a, b in _TROCAS.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120)
        self.cell(0, 5, _t(f"Relatório acadêmico gerado pelo app Trajetória térmica - Linha 711 "
                           f"em {self.gerado_em} | página {self.page_no()}"), align="C")


def gerar_pdf(res, ident, instrumentos, dados_simulados=False):
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
        pdf.set_font("Helvetica", "B", 8.5)
        for c, l in zip(cab, larg):
            pdf.cell(l, 6, _t(c), border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8.5)
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
        par("ATENÇÃO: valores de exemplo/simulados - não são medições reais.", 9.5, "B")
        pdf.set_text_color(0)

    titulo("1. Identificação")
    for k_, v in ident.items():
        par(f"{k_}: {v if str(v).strip() else '-'}")

    titulo("2. Base normativa e método")
    par("IBUTG com carga solar: 0,7 tbn + 0,2 tg + 0,1 tbs; sem carga solar: 0,7 tbn + 0,3 tg "
        "(NHO 06, Fundacentro, 3. ed., 2025; ISO 7243:2017). Medição conforme ISO 7726:2025. "
        "IBUTG da trajetória: média ponderada pelo tempo de permanência em cada etapa. "
        "Incerteza conforme JCGM 100:2008 (GUM). Climatização: ABNT NBR 15570:2021, item 33.2.2 "
        "(comparação indicativa). Referência ocupacional: NR-15, Anexo 3, Quadro 1 (Portaria "
        "SEPRT 1.359/2019), usada apenas como ordem de grandeza.")

    titulo("3. Instrumentação e incerteza tipo B")
    for k_, v in instrumentos.items():
        par(f"{k_}: {v}")

    titulo("4. Dados de entrada e resultados por etapa")
    t = res["tabela"]
    linhas = [[r["Etapa"][:28], f"{r['Duração (min)']:.0f}", "sim" if r["Carga solar"] else "não",
               f"{r['tbn']:.1f}", f"{r['tg']:.1f}",
               "-" if pd.isna(r["tbs"]) else f"{r['tbs']:.1f}",
               f"{r['IBUTG']:.1f} ± {r['U']:.1f}"] for _, r in t.iterrows()]
    tabela(["Etapa", "t (min)", "Sol", "tbn (°C)", "tg (°C)", "tbs (°C)", "IBUTG ± U (°C)"],
           linhas, [52, 16, 12, 20, 20, 20, 40])
    pdf.ln(2)
    par(f"IBUTG da trajetória (Eq. 3): {res['ibutg_traj']:.1f} ± {res['U_traj']:.1f} °C "
        f"(k = {res['k']:.0f}; duração total {res['T']:.0f} min).", 10, "B")
    par("A incerteza da trajetória soma linearmente as incertezas das etapas, porque o mesmo "
        "instrumento é usado em todas (erro sistemático correlacionado) - estimativa conservadora.",
        8.5, "I")

    if "dT" in res:
        d = res["dT"]
        titulo("5. Climatização do veículo (NBR 15570, item 33.2.2 - indicativo)")
        par(f"Te = {d['Te']:.1f} °C; Ti = {d['Ti']:.1f} °C; Delta T = {d['dT']:.1f} ± "
            f"{d['U']:.1f} °C; mínimo exigido = {d['dt_min']:.0f} °C quando Te > "
            f"{d['te_lim']:.0f} °C.")
        par(f"Situação: {d['situacao']}.", 10, "B")

    titulo("6. Conclusões")
    for c in conclusoes(res):
        par("- " + c)

    # Limitações + assinatura ficam juntas (evita página só com a assinatura)
    if pdf.get_y() + 55 > pdf.h - 15:
        pdf.add_page()
    titulo("7. Limitações")
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
