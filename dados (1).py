"""
dados.py - Dados de exemplo (SIMULADOS) e leitura de arquivos CSV.

ATENÇÃO: as medições do TCC só serão feitas no TCC II (2027). Até lá o app
usa dados SIMULADOS, gerados aqui com valores plausíveis para Cuiabá, apenas
para demonstrar os métodos. Eles NÃO são resultados de pesquisa.
Quando houver dados reais, basta carregá-los pelo app (CSV).
"""

import io
import re
import numpy as np
import pandas as pd

from normas import ibutg

# Série horária SINTÉTICA de temperatura do ar (°C), dia quente típico,
# horário local de Cuiabá (UTC-4). Formato igual ao que se extrai do INMET.
_HORAS = list(range(24))
_TEMP = [27.5, 27.0, 26.4, 25.9, 25.4, 25.0, 24.8, 25.9, 28.0, 30.2, 32.1, 33.6,
         34.9, 35.9, 36.6, 36.9, 36.5, 35.4, 33.6, 31.8, 30.6, 29.7, 28.9, 28.2]


def serie_horaria_exemplo():
    return pd.DataFrame({"hora": np.array(_HORAS, float), "temperatura": _TEMP})


def _hhmm(h):
    h = h % 24
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}"


def simular_viagem(serie, inicio_h=13.0, t_espera=15, t_veiculo=64, t_chegada=10,
                   seed=0, deslocamento=0.0, delta_local=0.8):
    """Simula o registro minuto a minuto de uma viagem da linha 711.

    Etapas: espera no ponto (ao sol) -> interior do veículo -> chegada
    (estação coberta). 'delta_local' é a diferença "verdadeira" entre a rota e
    a estação do INMET, que o app tenta recuperar pela Eq. 5.
    """
    rng = np.random.default_rng(seed)
    n = t_espera + t_veiculo + t_chegada
    minuto = np.arange(n + 1)
    hora = inicio_h + minuto / 60.0
    te = np.interp(hora, serie["hora"], serie["temperatura"]) + deslocamento + delta_local

    etapa = np.where(minuto < t_espera, "Espera no ponto",
                     np.where(minuto < t_espera + t_veiculo, "Interior do veículo",
                              "Chegada (estação)"))
    com_sol = etapa == "Espera no ponto"

    tbs = np.empty(n + 1)
    tbn = np.empty(n + 1)
    tg = np.empty(n + 1)
    ts = np.full(n + 1, np.nan)

    # --- espera ao sol (passagem de nuvem reduz o globo por alguns minutos)
    i_esp = np.where(etapa == "Espera no ponto")[0]
    nuvem = np.zeros(len(i_esp))
    if len(i_esp) > 6:
        k0 = rng.integers(2, len(i_esp) - 4)
        nuvem[k0:k0 + 3] = -4.0
    tbs[i_esp] = te[i_esp] + rng.normal(0, 0.3, len(i_esp))
    tbn[i_esp] = te[i_esp] - 8.0 + rng.normal(0, 0.3, len(i_esp))
    tg[i_esp] = te[i_esp] + 12.0 + nuvem + rng.normal(0, 0.8, len(i_esp))

    # --- interior do veículo: climatização + aberturas de porta + lotação
    i_vei = np.where(etapa == "Interior do veículo")[0]
    extra = np.zeros(len(i_vei))
    for k in range(1, len(i_vei)):
        porta = rng.uniform(1.0, 2.5) if rng.random() < 0.35 else 0.0
        extra[k] = 0.6 * extra[k - 1] + porta
    lotacao = np.where((np.arange(len(i_vei)) > 25) & (np.arange(len(i_vei)) < 45), 1.0, 0.0)
    ti = 26.0 + 0.3 * (te[i_vei] - 32.0) + extra + lotacao + rng.normal(0, 0.2, len(i_vei))
    ts[i_vei] = te[i_vei] + 5.0 + rng.normal(0, 1.2, len(i_vei))      # vidros aquecidos
    tbs[i_vei] = ti
    tbn[i_vei] = ti - 6.5 + rng.normal(0, 0.2, len(i_vei))
    tg[i_vei] = ti + 0.15 * (ts[i_vei] - ti) + rng.normal(0, 0.2, len(i_vei))

    # --- chegada: estação coberta, com ventiladores
    i_che = np.where(etapa == "Chegada (estação)")[0]
    tbs[i_che] = te[i_che] + rng.normal(0, 0.3, len(i_che))
    tbn[i_che] = te[i_che] - 8.8 + rng.normal(0, 0.3, len(i_che))
    tg[i_che] = te[i_che] + 1.2 + rng.normal(0, 0.4, len(i_che))

    df = pd.DataFrame({
        "minuto": minuto,
        "hora": hora,
        "horario": [_hhmm(h) for h in hora],
        "etapa": etapa,
        "com_sol": com_sol,
        "tbs": tbs.round(1),
        "tbn": tbn.round(1),
        "tg": tg.round(1),
        "ts_vidro": np.round(ts, 1),
    })
    df["IBUTG"] = np.where(df["com_sol"],
                           ibutg(df["tbn"], df["tg"], df["tbs"], com_sol=True),
                           ibutg(df["tbn"], df["tg"], com_sol=False)).round(2)
    return df


def simular_campanha(serie, n_dias=10, inicio_h=13.0, **kw):
    """Simula n_dias de coleta (Seção 3.3 do TCC: 10 dias úteis por faixa).

    Retorna (resumo por dia e etapa, todas as leituras minuto a minuto)."""
    rng = np.random.default_rng(1234)
    resumo, todas = [], []
    for d in range(1, n_dias + 1):
        desloc = rng.normal(0, 1.0)
        df = simular_viagem(serie, inicio_h=inicio_h, seed=100 + d, deslocamento=desloc, **kw)
        df["dia"] = d
        todas.append(df)
        for etapa, g in df.groupby("etapa", sort=False):
            resumo.append({"dia": d, "etapa": etapa,
                           "IBUTG médio": g["IBUTG"].mean(),
                           "duração (min)": len(g)})
    return pd.DataFrame(resumo), pd.concat(todas, ignore_index=True)


def csv_exemplo_bytes():
    """CSV simples (hora local; temperatura) para o usuário ver o formato."""
    s = serie_horaria_exemplo()
    s["hora"] = [f"{int(h):02d}:00" for h in s["hora"]]
    return s.to_csv(index=False, sep=";", decimal=",").encode("utf-8")


def _hora_de_texto(v):
    """'1300 UTC' -> 13 ; '13:30' -> 13,5 ; '13' ou '13,0' -> 13."""
    s = str(v).strip().lower()
    m = re.match(r"^(\d{1,2})[:h](\d{2})", s)
    if m:                                   # 13:30
        return int(m.group(1)) + int(m.group(2)) / 60.0
    m = re.match(r"^(\d{3,4})\s*(utc)?$", s)
    if m:                                   # 1300 UTC
        d = m.group(1)
        return int(d[:-2]) + int(d[-2:]) / 60.0
    try:                                    # 13 ou 13,0
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def ler_csv_temperatura(conteudo, fuso_utc=-4):
    """Lê (a) o CSV de dados históricos do INMET (cabeçalho com metadados,
    separador ';', vírgula decimal, 'Hora UTC') ou (b) um CSV simples com
    colunas 'hora' (local) e 'temperatura'.

    Retorna DataFrame com colunas: data, hora (local), temperatura.
    Leitor do formato INMET feito com base no layout público dos arquivos;
    confira o resultado com o arquivo original.
    """
    if isinstance(conteudo, bytes):
        try:
            texto = conteudo.decode("utf-8")
        except UnicodeDecodeError:
            texto = conteudo.decode("latin-1")
    else:
        texto = conteudo
    linhas = texto.splitlines()
    idx = next((i for i, l in enumerate(linhas[:40]) if "hora" in l.lower()), None)
    if idx is None:
        raise ValueError("Não encontrei uma coluna de hora no arquivo.")
    sep = ";" if ";" in linhas[idx] else ","
    df = pd.read_csv(io.StringIO("\n".join(linhas[idx:])), sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    low = {c: c.lower() for c in df.columns}

    col_hora = next(c for c in df.columns if "hora" in low[c])
    utc = "utc" in low[col_hora]
    col_temp = next((c for c in df.columns if "bulbo seco" in low[c]), None)
    if col_temp is None:
        col_temp = next((c for c in df.columns if low[c].startswith("temp")), None)
    if col_temp is None:
        raise ValueError("Não encontrei a coluna de temperatura.")
    col_data = next((c for c in df.columns if low[c].startswith("data")), None)

    temp = pd.to_numeric(df[col_temp].str.replace(",", ".", regex=False), errors="coerce")
    hora = df[col_hora].map(_hora_de_texto)
    if col_data is not None:
        data = pd.to_datetime(df[col_data], errors="coerce")
    else:
        data = pd.Series(pd.Timestamp("2000-01-01"), index=df.index)

    dt = data + pd.to_timedelta(hora, unit="h")
    if utc:
        dt = dt + pd.to_timedelta(fuso_utc, unit="h")
    out = pd.DataFrame({"dt": dt, "temperatura": temp}).dropna()
    out = out[(out["temperatura"] > -50) & (out["temperatura"] < 60)]   # remove -9999 etc.
    out["data"] = out["dt"].dt.strftime("%Y-%m-%d")
    out["hora"] = out["dt"].dt.hour + out["dt"].dt.minute / 60.0
    out = out.sort_values("dt").drop_duplicates("dt")
    return out[["data", "hora", "temperatura"]].reset_index(drop=True)


COLUNAS_VIAGEM = ["horario", "etapa", "com_sol", "tbs", "tbn", "tg", "ts_vidro"]
ETAPAS = ["Espera no ponto", "Interior do veículo", "Chegada (estação)"]


def viagem_para_csv(df):
    """Exporta uma viagem no formato aceito por ler_csv_viagem (modelo para o TCC II)."""
    out = df[COLUNAS_VIAGEM].copy()
    out["com_sol"] = out["com_sol"].astype(int)
    return out.to_csv(index=False, sep=";", decimal=",").encode("utf-8")


def ler_csv_viagem(conteudo):
    """Lê medições reais de uma viagem (registro a cada minuto).

    Colunas: horario (HH:MM), etapa (Espera no ponto / Interior do veículo /
    Chegada (estação)), com_sol (1/0), tbs, tbn, tg, ts_vidro (opcional).
    """
    texto = conteudo.decode("utf-8-sig", errors="replace") if isinstance(conteudo, bytes) else conteudo
    sep = ";" if texto.splitlines()[0].count(";") >= texto.splitlines()[0].count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    faltando = [c for c in ["horario", "etapa", "com_sol", "tbs", "tbn", "tg"] if c not in df.columns]
    if faltando:
        raise ValueError(f"Faltam as colunas: {', '.join(faltando)}")
    for c in ["tbs", "tbn", "tg", "ts_vidro"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
        else:
            df[c] = np.nan
    df["com_sol"] = df["com_sol"].astype(str).str.strip().str.lower().isin(["1", "true", "sim", "s"])
    df["hora"] = df["horario"].map(_hora_de_texto)
    df = df.dropna(subset=["hora", "tbs", "tbn", "tg"]).reset_index(drop=True)
    # viagem que passa da meia-noite
    df["hora"] = df["hora"] + 24 * (df["hora"].diff().fillna(0) < -12).cumsum()
    df["minuto"] = np.round((df["hora"] - df["hora"].iloc[0]) * 60).astype(int)
    df["etapa"] = df["etapa"].str.strip()
    df["IBUTG"] = np.where(df["com_sol"],
                           ibutg(df["tbn"], df["tg"], df["tbs"], com_sol=True),
                           ibutg(df["tbn"], df["tg"], com_sol=False)).round(2)
    return df
