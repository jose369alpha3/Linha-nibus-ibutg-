"""
normas.py - Parâmetros normativos usados no app, com a fonte de cada um.

Somente valores numéricos e equações, com citação. O texto das normas
(ABNT, ISO) é protegido e não é reproduzido aqui.
Edições conferidas em nov/2026.
"""

import numpy as np

# -----------------------------------------------------------------------------
# IBUTG - NHO 06 (Fundacentro, 3. ed., 2025) e ISO 7243:2017
# -----------------------------------------------------------------------------
COEF_COM_SOL = {"tbn": 0.7, "tg": 0.2, "tbs": 0.1}   # Eq. 1 do TCC
COEF_SEM_SOL = {"tbn": 0.7, "tg": 0.3}               # Eq. 2 do TCC


def ibutg(tbn, tg, tbs=None, com_sol=True):
    """IBUTG [°C]. Com carga solar: 0,7 tbn + 0,2 tg + 0,1 tbs.
    Sem carga solar: 0,7 tbn + 0,3 tg."""
    tbn = np.asarray(tbn, float)
    tg = np.asarray(tg, float)
    if com_sol:
        return 0.7 * tbn + 0.2 * tg + 0.1 * np.asarray(tbs, float)
    return 0.7 * tbn + 0.3 * tg


# -----------------------------------------------------------------------------
# ABNT NBR 15570:2021 (5. ed.) - item 33.2.2
# -----------------------------------------------------------------------------
NBR15570_TE_LIMIAR = 32.0     # °C: acima disso o requisito se aplica
NBR15570_DT_MINIMO = 8.0      # °C: diferença mínima Te - Ti

# -----------------------------------------------------------------------------
# NR-15, Anexo 3 (redação Portaria SEPRT 1.359/2019), Quadro 1 - trecho
# M [W] x IBUTG_MÁX [°C]. Usado só como ORDEM DE GRANDEZA (não se aplica a
# passageiros; item 1.1.1 exclui atividades a céu aberto sem fonte artificial
# de calor). Valores transcritos do PDF oficial (gov.br) - conferir.
# Quadro 2: sentado em repouso = 100 W; em pé em repouso = 126 W.
# -----------------------------------------------------------------------------
NR15_QUADRO1_M = np.array([100, 102, 104, 106, 108, 110, 112, 115, 117, 119,
                           122, 124, 127, 129, 132, 135, 137, 140], float)
NR15_QUADRO1_IBUTG = np.array([33.7, 33.6, 33.5, 33.4, 33.3, 33.2, 33.1, 33.0, 32.9,
                               32.8, 32.7, 32.6, 32.5, 32.4, 32.3, 32.2, 32.1, 32.0])
M_SENTADO_REPOUSO = 100.0
M_EM_PE_REPOUSO = 126.0

# -----------------------------------------------------------------------------
# ISO 7726:2025 (3. ed.) - altura de medição (abdômen)
# -----------------------------------------------------------------------------
ALTURA_EM_PE_M = 1.1
ALTURA_SENTADO_M = 0.6

# -----------------------------------------------------------------------------
# GUM - JCGM 100:2008
# -----------------------------------------------------------------------------
K_ABRANGENCIA = 2.0   # ~95 %

REFERENCIAS = [
    ("NHO 06 (Fundacentro, 3. ed., 2025)", "Cálculo do IBUTG e média ponderada no tempo"),
    ("ISO 7243:2017", "Índice WBGT (IBUTG) - com e sem carga solar"),
    ("ISO 7726:2025 (3. ed.)", "Instrumentos e procedimento de medição; altura de 1,1 m / 0,6 m"),
    ("ABNT NBR 15570:2021 (5. ed.), item 33.2.2", "Te > 32 °C => Te - Ti >= 8 °C (comparação indicativa)"),
    ("NR-15, Anexo 3 (Portaria SEPRT 1.359/2019), Quadros 1 e 2",
     "Limites ocupacionais usados só como ordem de grandeza"),
    ("JCGM 100:2008 (GUM)", "Incerteza tipo A, tipo B, combinada e expandida (k = 2)"),
    ("Ruggiero & Lopes, Cálculo Numérico, 2. ed.", "Métodos numéricos (caps. 2, 3, 5, 6 e 7)"),
]
