# ColheAí — IA para a meta da agricultura familiar no PNAE

**Global Solution 2026/2 · CitizenTech · Artificial Intelligence & Chatbot**
Since Young · Turma 1TDSPV-2026 · FIAP · Prof. Alfonso José Trindade Rodriguez

| Integrante | RM |
|---|---|
| Artur Bianchini Esteves (representante) | RM569450 |
| João Vitor Oliveira Fontenele Martins | RM570783 |

## Objetivo

A Lei 15.226/2025 obriga cada prefeitura a gastar **45%** do repasse federal do PNAE (merenda escolar) comprando
direto da agricultura familiar, a partir de 2026. Antes eram 30%, e muitos municípios não cumpriam nem isso.
O ColheAí usa dois modelos de aprendizado de máquina, treinados com **dados públicos e reais** do FNDE e do IBGE
para os **645 municípios de São Paulo**:

1. **Classificação (Regressão Logística):** risco de o município ficar abaixo da meta no próximo período.
2. **Agrupamento (K-Means, k = 4):** perfil do município e abordagem comercial do ColheAí.

Os dois ficam disponíveis numa **API REST em Flask**, consultada pelo código IBGE do município, a mesma chave
(`cd_ibge_municipio`) que a tabela `T_CLH_ENTIDADE_EXECUTORA` do banco do projeto usa.

## Resultados

| Modelo | Métrica | Valor |
|---|---|---|
| Regressão Logística (escolhida) | F1 / precisão / recall (classe "abaixo da meta") | 0,774 / 0,757 / 0,792 |
| | AUC-ROC (teste) · validação cruzada 5x | 0,813 · 0,793 ± 0,027 |
| | AUC fora do tempo (histórico 2017–2019 → resultado 2022) | 0,763 |
| Random Forest (comparação) | F1 / AUC (teste) | 0,785 / 0,795, empate técnico: ficou o modelo mais simples |
| K-Means | silhueta (k = 4) | 0,226 |

## Estrutura

```
IA_SinceYoung_1TDSPV/
├── README.md                       este arquivo
├── integrantes.txt                 RM, nome completo, turma e explicações
├── render.yaml                     publicação da API no Render
├── dados/
│   ├── fnde_pnae_agricultura_familiar_sp_2011_2022.csv   base bruta FNDE (município x ano)
│   ├── municipios_sp_base_bruta.csv                      base bruta por município (FNDE + IBGE)
│   ├── municipios_sp_base_normalizada.csv                base normalizada Min-Max (0 a 1)
│   ├── coleta_dados_publicos.py                          script que baixa e monta as bases brutas
│   └── LEIA-ME_DADOS.txt                                 fontes, data de acesso e dicionário
├── notebooks/
│   ├── ColheAi_IA_Modelos.ipynb    EDA → preparação → treino → avaliação → exportação (já executado)
│   └── figuras/                    gráficos gerados pelo notebook
├── modelos/
│   ├── modelo_risco_meta.joblib    pacote {modelo, colunas, parametros_minmax, metricas, versao, equipe, ...}
│   └── modelo_perfil_kmeans.joblib idem, para o K-Means
└── api/
    ├── api_colheai_ia.py           API Flask
    ├── requirements.txt            scikit-learn pinado na versão do treino (1.9.1)
    └── .env.example                variáveis de ambiente (sem valores)
```

## Tecnologias

Python 3.13 · pandas · numpy · matplotlib · scikit-learn 1.9.1 · joblib · Flask · flask-cors · gunicorn · Render.

## Como executar

```bash
# 1. dependências
pip install -r api/requirements.txt matplotlib jupyter

# 2. (opcional) baixar de novo os dados públicos, os CSV já acompanham a entrega
cd dados && python coleta_dados_publicos.py && cd ..

# 3. (opcional) reexecutar o notebook, que regrava a base normalizada e os .joblib
cd notebooks && jupyter nbconvert --to notebook --execute --inplace ColheAi_IA_Modelos.ipynb && cd ..

# 4. subir a API
python api/api_colheai_ia.py          # http://localhost:5000
```

## Endpoints da API

| Método e rota | O que faz |
|---|---|
| `GET /` | health check (200 mesmo se um modelo falhar) |
| `GET /api/status` | versão, métricas e campos aceitos dos dois modelos |
| `POST /api/prever-risco` | variáveis do município → probabilidade e faixa de risco (BAIXO, MEDIO, ALTO) |
| `POST /api/perfil-municipio` | variáveis de perfil → grupo do K-Means e abordagem |
| `GET /api/municipio/<cd_ibge>` | busca pelo código IBGE e devolve os dois resultados |

```bash
curl http://localhost:5000/api/municipio/3550308

curl -X POST http://localhost:5000/api/prever-risco -H "Content-Type: application/json" \
  -d '{"qt_populacao": 66970, "vl_densidade_hab_km2": 430.29, "pc_populacao_rural": 10.37,
       "vl_pib_per_capita": 20683.01, "estab_af_por_mil_hab": 0.866, "vl_transferido_medio": 120000,
       "pc_af_media_3anos": 5.0, "qt_anos_sem_compra_af_3anos": 2, "qt_anos_cumpriu_30_6anos": 0,
       "nm_regiao_intermediaria": "SAO_PAULO"}'
```

## Links

- API publicada (Render): `https://colheai-api-ia.onrender.com` *(confirmar após o deploy)*
- Repositório: *(preencher após criar)*

O plano gratuito do Render hiberna após 15 minutos sem uso; a primeira chamada pode levar cerca de 50 segundos.

## Sobre os dados (honestidade)

- **Nada foi simulado.** FNDE (PNAE: Dados da Agricultura Familiar, 2011–2022) e IBGE (Censo 2022, PIB 2021,
  Censo Agropecuário 2017), acessados em 17/11/2026.
- O FNDE só publica um painel Power BI; o script consulta a API pública que o próprio painel usa.
- Corrigimos 94 linhas com código IBGE de município homônimo de outro estado.
- 2022 é preliminar (aviso do FNDE) e 2020–2021 são anos de pandemia: ficam fora do rótulo.
- O rótulo usa a meta de **30%** (a vigente no período dos dados); para a meta de 45% o risco é um piso.

## Uso de IA

Este trabalho foi desenvolvido com apoio do **Claude (Anthropic, claude.ai)** como assistente técnico.
O Claude buscou as fontes, escreveu o script de coleta, o notebook e a API seguindo o modelo aprovado no
challenge (Sprint 4) e os padrões do professor. O grupo definiu o problema e o escopo, revisou as decisões
(alvo de 30% em vez de 45%, uso da Regressão Logística em vez do Random Forest, exclusão de 2022 do treino)
e validou executando o notebook do início ao fim, testando a API com `curl` e conferindo as métricas
contra a rubrica. O registro completo, com prompts e revisão crítica, está no documento de SEBM.

## Contato

Artur Bianchini Esteves · RM569450 · Since Young · FIAP
