"""
metodos.py - Núcleo numérico do app (implementado do zero, sem scipy).

Referência principal:
RUGGIERO, M. A. G.; LOPES, V. L. R. Cálculo Numérico: aspectos teóricos e
computacionais. 2. ed. São Paulo: Makron Books, 1996.

Cada função indica o capítulo/seção do livro em que o método é descrito.
O numpy é usado só para guardar vetores e fazer contas elementares.
"""

import math
import numpy as np


# =============================================================================
# CAPÍTULO 3 - SISTEMAS LINEARES
# =============================================================================
def gauss_pivoteamento(A, b):
    """Eliminação de Gauss com pivoteamento parcial (Ruggiero, seção 3.2.2).

    Resolve A x = b. Usada pela spline cúbica (cap. 5) e pelos quadrados
    mínimos (cap. 6), que recaem em sistemas lineares.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)])  # matriz aumentada [A | b]

    # Fase 1: triangularização
    for k in range(n - 1):
        # pivoteamento parcial: traz para a linha k o maior |a_ik|, i >= k
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < 1e-14:
            raise ValueError("Matriz singular (pivô nulo).")
        if p != k:
            M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]          # multiplicador
            M[i, k:] = M[i, k:] - m * M[k, k:]

    if abs(M[n - 1, n - 1]) < 1e-14:
        raise ValueError("Matriz singular (pivô nulo).")

    # Fase 2: substituição retroativa
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
    return x


# =============================================================================
# CAPÍTULO 5 - INTERPOLAÇÃO
# =============================================================================
def spline_linear(xs, ys, x):
    """Spline linear interpolante (Ruggiero, seção 5.7.1).

    Em cada intervalo [x_i, x_{i+1}] liga os pontos por uma reta.
    """
    escalar = np.ndim(x) == 0
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    x = np.atleast_1d(np.asarray(x, float))
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    h = xs[i + 1] - xs[i]
    y = ys[i] * (xs[i + 1] - x) / h + ys[i + 1] * (x - xs[i]) / h
    return float(y[0]) if escalar else y


def spline_linear_derivada(xs, ys, x):
    """Derivada da spline linear: inclinação do segmento que contém x."""
    escalar = np.ndim(x) == 0
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    x = np.atleast_1d(np.asarray(x, float))
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    d = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return float(d[0]) if escalar else d


class SplineCubicaNatural:
    """Spline cúbica interpolante natural (Ruggiero, seção 5.7.2).

    Incógnitas: g_i = S''(x_i). Condição natural: g_0 = g_n = 0.
    Para i = 1..n-1:
        h_{i-1} g_{i-1} + 2(h_{i-1} + h_i) g_i + h_i g_{i+1}
            = 6 [ (y_{i+1} - y_i)/h_i - (y_i - y_{i-1})/h_{i-1} ]
    O sistema (tridiagonal) é resolvido por Eliminação de Gauss (cap. 3).
    """

    def __init__(self, xs, ys):
        self.xs = np.asarray(xs, float)
        self.ys = np.asarray(ys, float)
        n = len(self.xs) - 1
        if n < 2:
            raise ValueError("A spline cúbica precisa de pelo menos 3 pontos.")
        h = np.diff(self.xs)
        if np.any(h <= 0):
            raise ValueError("Os x devem ser estritamente crescentes.")
        self.h = h

        # monta o sistema (n-1) x (n-1) para g_1 ... g_{n-1}
        A = np.zeros((n - 1, n - 1))
        b = np.zeros(n - 1)
        for i in range(1, n):
            r = i - 1
            if r > 0:
                A[r, r - 1] = h[i - 1]
            A[r, r] = 2 * (h[i - 1] + h[i])
            if r < n - 2:
                A[r, r + 1] = h[i]
            b[r] = 6 * ((self.ys[i + 1] - self.ys[i]) / h[i]
                        - (self.ys[i] - self.ys[i - 1]) / h[i - 1])
        self.A, self.b = A, b
        g_int = gauss_pivoteamento(A, b)
        self.g = np.concatenate([[0.0], g_int, [0.0]])

    def _intervalo(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        i = np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2)
        return x, i

    def __call__(self, x):
        """S(x)."""
        escalar = np.ndim(x) == 0
        x, i = self._intervalo(x)
        xs, ys, g, h = self.xs, self.ys, self.g, self.h[i]
        a = xs[i + 1] - x
        c = x - xs[i]
        s = (g[i] * a**3 / (6 * h) + g[i + 1] * c**3 / (6 * h)
             + (ys[i] / h - g[i] * h / 6) * a
             + (ys[i + 1] / h - g[i + 1] * h / 6) * c)
        return float(s[0]) if escalar else s

    def derivada(self, x):
        """S'(x) - usada pelo Método de Newton (cap. 2)."""
        escalar = np.ndim(x) == 0
        x, i = self._intervalo(x)
        xs, ys, g, h = self.xs, self.ys, self.g, self.h[i]
        a = xs[i + 1] - x
        c = x - xs[i]
        d = (-g[i] * a**2 / (2 * h) + g[i + 1] * c**2 / (2 * h)
             - (ys[i] / h - g[i] * h / 6)
             + (ys[i + 1] / h - g[i + 1] * h / 6))
        return float(d[0]) if escalar else d


def newton_diferencas_divididas(xs, ys):
    """Coeficientes do polinômio interpolador na forma de Newton
    (Ruggiero, seção 5.3.3): d_k = f[x_0, ..., x_k]."""
    xs = np.asarray(xs, float)
    d = np.array(ys, dtype=float)
    n = len(xs)
    for k in range(1, n):
        d[k:] = (d[k:] - d[k - 1:-1]) / (xs[k:] - xs[:n - k])
    return d


def avaliar_newton(coef, xs, x):
    """Avalia p_n(x) na forma de Newton (esquema de Horner)."""
    escalar = np.ndim(x) == 0
    xs = np.asarray(xs, float)
    x = np.atleast_1d(np.asarray(x, float))
    p = np.full_like(x, coef[-1])
    for k in range(len(coef) - 2, -1, -1):
        p = p * (x - xs[k]) + coef[k]
    return float(p[0]) if escalar else p


# =============================================================================
# CAPÍTULO 2 - ZEROS REAIS DE FUNÇÕES REAIS
# =============================================================================
def isolar_raizes(f, a, b, n_sub=200):
    """Fase I - isolamento das raízes (Ruggiero, seção 2.2).

    Divide [a, b] em subintervalos e guarda os que têm troca de sinal
    (Teorema de Bolzano: f(a_k) f(b_k) < 0 => existe raiz em [a_k, b_k]).
    """
    x = np.linspace(a, b, n_sub + 1)
    fx = np.array([float(f(xi)) for xi in x])
    intervalos = []
    for k in range(n_sub):
        if fx[k] == 0:
            intervalos.append((x[k], x[k]))
        elif fx[k] * fx[k + 1] < 0:
            intervalos.append((x[k], x[k + 1]))
    return intervalos


def bissecao(f, a, b, eps=1e-6, max_iter=100):
    """Método da Bissecção (Ruggiero, seção 2.3.2-I).

    Critério de parada: (b - a) < eps  ou  |f(x)| < eps.
    """
    fa, fb = float(f(a)), float(f(b))
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) têm o mesmo sinal: intervalo sem raiz isolada.")
    tabela = []
    x = (a + b) / 2
    for k in range(1, max_iter + 1):
        x = (a + b) / 2
        fx = float(f(x))
        tabela.append({"k": k, "a": a, "b": b, "x_k": x, "f(x_k)": fx, "b - a": b - a})
        if (b - a) < eps or abs(fx) < eps:
            return {"raiz": x, "iteracoes": k, "tabela": tabela, "convergiu": True}
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    return {"raiz": x, "iteracoes": max_iter, "tabela": tabela, "convergiu": False}


def newton_raphson(f, df, x0, eps=1e-6, max_iter=50):
    """Método de Newton-Raphson (Ruggiero, seção 2.3.2-IV).

    x_{k+1} = x_k - f(x_k) / f'(x_k)
    Critério de parada: |x_{k+1} - x_k| < eps  ou  |f(x_{k+1})| < eps.
    """
    tabela = []
    x = float(x0)
    for k in range(1, max_iter + 1):
        fx = float(f(x))
        dfx = float(df(x))
        if abs(dfx) < 1e-14:
            raise ValueError("Derivada nula: Newton não pode continuar.")
        x_novo = x - fx / dfx
        f_novo = float(f(x_novo))
        tabela.append({"k": k, "x_k": x, "f(x_k)": fx, "f'(x_k)": dfx,
                       "x_k+1": x_novo, "|x_k+1 - x_k|": abs(x_novo - x)})
        if abs(x_novo - x) < eps or abs(f_novo) < eps:
            return {"raiz": x_novo, "iteracoes": k, "tabela": tabela, "convergiu": True}
        x = x_novo
    return {"raiz": x, "iteracoes": max_iter, "tabela": tabela, "convergiu": False}


def secante(f, x0, x1, eps=1e-6, max_iter=50):
    """Método da Secante (Ruggiero, seção 2.3.2-V).

    x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
    """
    tabela = []
    xa, xb = float(x0), float(x1)
    fa, fb = float(f(xa)), float(f(xb))
    for k in range(1, max_iter + 1):
        if abs(fb - fa) < 1e-14:
            raise ValueError("f(x_k) = f(x_k-1): secante não pode continuar.")
        x_novo = xb - fb * (xb - xa) / (fb - fa)
        f_novo = float(f(x_novo))
        tabela.append({"k": k, "x_k-1": xa, "x_k": xb, "x_k+1": x_novo,
                       "f(x_k+1)": f_novo, "|x_k+1 - x_k|": abs(x_novo - xb)})
        if abs(x_novo - xb) < eps or abs(f_novo) < eps:
            return {"raiz": x_novo, "iteracoes": k, "tabela": tabela, "convergiu": True}
        xa, fa, xb, fb = xb, fb, x_novo, f_novo
    return {"raiz": xb, "iteracoes": max_iter, "tabela": tabela, "convergiu": False}


# =============================================================================
# CAPÍTULO 7 - INTEGRAÇÃO NUMÉRICA
# =============================================================================
def trapezios_repetida(y, h):
    """Regra dos Trapézios Repetida (Ruggiero, seção 7.2.2), dados tabelados
    com passo constante h:  I = h/2 [y_0 + 2(y_1 + ... + y_{m-1}) + y_m]."""
    y = np.asarray(y, float)
    if len(y) < 2:
        raise ValueError("São necessários pelo menos 2 pontos.")
    return h / 2 * (y[0] + 2 * y[1:-1].sum() + y[-1])


def simpson_repetida(y, h):
    """Regra 1/3 de Simpson Repetida (Ruggiero, seção 7.2.4).

    I = h/3 [y_0 + 4(y_1 + y_3 + ...) + 2(y_2 + y_4 + ...) + y_m],
    exige m (número de subintervalos) par. Se m for ímpar, aplica Simpson
    nos m-1 primeiros e Trapézio no último (e avisa).
    """
    y = np.asarray(y, float)
    m = len(y) - 1
    if m < 2:
        return trapezios_repetida(y, h), True
    ajuste = False
    if m % 2 == 1:
        ajuste = True
        cauda = h / 2 * (y[-2] + y[-1])
        y = y[:-1]
    s = h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])
    if ajuste:
        s += cauda
    return s, ajuste


# =============================================================================
# CAPÍTULO 6 - AJUSTE DE CURVAS PELO MÉTODO DOS QUADRADOS MÍNIMOS
# =============================================================================
def minimos_quadrados_polinomio(x, y, grau=1):
    """Quadrados mínimos, caso discreto (Ruggiero, seção 6.2.1).

    Ajusta phi(x) = a_0 + a_1 x + ... + a_g x^g. As equações normais
    (sum_k x_k^{i+j}) a_j = sum_k y_k x_k^i  são resolvidas por Gauss (cap. 3).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = int(grau)
    A = np.array([[np.sum(x ** (i + j)) for j in range(g + 1)] for i in range(g + 1)])
    b = np.array([np.sum(y * x ** i) for i in range(g + 1)])
    coef = gauss_pivoteamento(A, b)
    y_aj = sum(coef[i] * x ** i for i in range(g + 1))
    residuos = y - y_aj
    sq_res = float(np.sum(residuos ** 2))
    sq_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sq_res / sq_tot if sq_tot > 0 else float("nan")
    return {"coef": coef, "A": A, "b": b, "r2": r2,
            "residuos": residuos, "soma_quad_residuos": sq_res}


def avaliar_polinomio(coef, x):
    x = np.asarray(x, float)
    return sum(c * x ** i for i, c in enumerate(coef))


# =============================================================================
# INCERTEZA DE MEDIÇÃO - GUM (JCGM 100:2008); Eqs. 6 a 14 do TCC
# =============================================================================
def incerteza_tipo_b(exatidao, resolucao):
    """Eq. 6: u_B = sqrt( (a/√3)² + (r/(2√3))² ), distribuição retangular."""
    return math.sqrt((exatidao / math.sqrt(3)) ** 2 + (resolucao / (2 * math.sqrt(3))) ** 2)


def incerteza_ibutg(u_tbn, u_tg, u_tbs=0.0, com_sol=True):
    """Eq. 8: propagação (Eq. 7) para o IBUTG; os coeficientes de sensibilidade
    são os próprios pesos da equação, pois o índice é linear."""
    if com_sol:
        return math.sqrt((0.7 * u_tbn) ** 2 + (0.2 * u_tg) ** 2 + (0.1 * u_tbs) ** 2)
    return math.sqrt((0.7 * u_tbn) ** 2 + (0.3 * u_tg) ** 2)


def estatistica_tipo_a(amostras):
    """Eqs. 10-12: média, desvio padrão amostral (n-1) e u_A = s/√n."""
    x = np.asarray(amostras, float)
    n = len(x)
    media = float(x.mean())
    s = float(math.sqrt(np.sum((x - media) ** 2) / (n - 1))) if n > 1 else float("nan")
    return {"n": n, "media": media, "s": s, "u_A": s / math.sqrt(n) if n > 1 else float("nan")}


def incerteza_expandida(u_a, u_b, k=2.0):
    """Eqs. 13-14: u_c = sqrt(u_A² + u_B²);  U = k u_c."""
    u_c = math.sqrt(u_a ** 2 + u_b ** 2)
    return {"u_c": u_c, "U": k * u_c}
