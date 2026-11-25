# Trajetória térmica · Linha 711

Projeto individual de **Cálculo Numérico** (UFMT): aplicação de métodos numéricos à proposta
de TCC *"Condições térmicas e exposição ao calor de usuários do transporte coletivo em
Cuiabá-MT: estudo de caso na linha 711"*. Autor: José Viana Junior.

> ⚠️ Enquanto não há medições (TCC II, 2027), o app usa **dados simulados**, apenas para
> demonstrar os métodos. Eles não são resultados de pesquisa.

## Métodos (Ruggiero & Lopes, *Cálculo Numérico*, 2. ed.)

| Módulo | Método | Capítulo | Uso no problema |
|---|---|---|---|
| 1 | Spline linear e spline cúbica natural; polinômio de Newton (Runge) | 5 | Te(t) a partir da série horária do INMET (Eq. 5) |
| 1, 4 | Eliminação de Gauss com pivoteamento parcial | 3 | Sistema da spline cúbica e equações normais |
| 2 | Bissecção, Newton-Raphson, Secante (isolamento + refinamento) | 2 | Instantes em que Te(t) = 32 °C (NBR 15570, 33.2.2) |
| 3 | Trapézios repetida e 1/3 de Simpson repetida | 7 | IBUTG médio no tempo (Eq. 3) e intervalo de registro |
| 4 | Quadrados mínimos (caso discreto) | 6 | IBUTG interno × temperatura dos vidros |
| 5 | GUM: incerteza tipo A, tipo B, combinada e expandida | — | Resultado IBUTG ± U por etapa |

Todos os métodos estão implementados do zero em `metodos.py` e foram conferidos contra
NumPy/SciPy (diferenças da ordem de 1e-10 ou menores).

## Relatório técnico

Na aba **📝 Relatório** o usuário digita as medições de cada etapa (tbn, tg, tbs, duração,
carga solar) e Te/Ti, e o app gera o IBUTG por etapa e da trajetória (com incerteza), a
verificação indicativa da NBR 15570 e um **PDF**. O relatório é acadêmico: não é laudo de
insalubridade (NR-15) nem ensaio de conformidade.

## Arquivos

- `app.py` – interface (Streamlit)
- `metodos.py` – métodos numéricos
- `normas.py` – parâmetros normativos com a fonte (NHO 06, ISO 7243, ISO 7726, NBR 15570, NR-15, GUM)
- `dados.py` – dados simulados e leitura de CSV (INMET e medições da viagem)
- `relatorio.py` – avaliação a partir de valores digitados e relatório técnico em PDF
- `requirements.txt` – bibliotecas

## Rodar no computador

```
pip install -r requirements.txt
streamlit run app.py
```

## Normas

Só os parâmetros numéricos são usados, com citação. O texto das normas ABNT/ISO é
protegido e não é reproduzido. Edições conferidas em nov/2026.
