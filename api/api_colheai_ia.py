# -*- coding: utf-8 -*-
"""
================================================================================
COLHEAI - API REST DOS MODELOS DE IA (Flask)
Global Solution 2026/2 - CitizenTech - Artificial Intelligence & Chatbot
Since Young - Turma 1TDSPV-2026 - FIAP
--------------------------------------------------------------------------------
Artur Bianchini Esteves                RM569450   REPRESENTANTE
Joao Vitor Oliveira Fontenele Martins  RM570783
--------------------------------------------------------------------------------
O QUE ESTA API FAZ

Expoe os DOIS modelos treinados no notebook notebooks/ColheAi_IA_Modelos.ipynb:

  1) modelo_risco_meta.joblib (classificacao - Regressao Logistica)
     Probabilidade de o municipio ficar abaixo da meta de compra da
     agricultura familiar no PNAE no proximo periodo.
  2) modelo_perfil_kmeans.joblib (agrupamento - K-Means)
     Perfil do municipio entre 4 grupos, com a abordagem comercial do ColheAi.

O cliente NAO precisa conhecer o pre-processamento. A API reproduz o log10, o
preenchimento pela mediana, o one-hot da regiao e o Min-Max com os parametros
gravados DENTRO dos arquivos .joblib. Se a normalizacao usasse outro minimo ou
maximo, a previsao sairia errada sem nenhum erro visivel.

Os dados sao publicos e reais (FNDE e IBGE, acesso em 17/11/2026). A API nao
acessa banco de dados e nao usa nenhuma credencial.

--------------------------------------------------------------------------------
ENDPOINTS

  GET  /                        Health check. Responde 200 mesmo que um modelo
                                falhe ao carregar (o Render precisa disso).
  GET  /api/status              Versao, metricas e campos aceitos dos 2 modelos.
  POST /api/prever-risco        Variaveis de um municipio -> risco da meta.
  POST /api/perfil-municipio    Variaveis de perfil -> grupo do K-Means.
  GET  /api/municipio/<cd_ibge> Busca o municipio pelo codigo IBGE (o mesmo
                                cd_ibge_municipio da T_CLH_ENTIDADE_EXECUTORA)
                                e devolve os DOIS resultados. E a rota do front.

--------------------------------------------------------------------------------
COMO RODAR LOCALMENTE

    pip install -r requirements.txt
    python api_colheai_ia.py            (sobe em http://localhost:5000)

EXEMPLOS
    curl http://localhost:5000/api/municipio/3550308

    curl -X POST http://localhost:5000/api/prever-risco \\
         -H "Content-Type: application/json" \\
         -d '{"qt_populacao": 66970, "vl_densidade_hab_km2": 430.29,
              "pc_populacao_rural": 10.37, "vl_pib_per_capita": 20683.01,
              "estab_af_por_mil_hab": 0.866, "vl_transferido_medio": 120000,
              "pc_af_media_3anos": 5.0, "qt_anos_sem_compra_af_3anos": 2,
              "qt_anos_cumpriu_30_6anos": 0, "nm_regiao_intermediaria": "SAO_PAULO"}'
================================================================================
"""

import math
import os
import re
import unicodedata
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, request

try:
    from flask_cors import CORS
    TEM_CORS = True
except ImportError:
    TEM_CORS = False

# ==============================================================================
# CONFIGURACAO (tudo por variavel de ambiente, com padrao seguro)
# ==============================================================================
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.environ.get("PASTA_MODELOS", os.path.join(PASTA_BASE, "..", "modelos"))
ARQUIVO_RISCO = os.environ.get("ARQUIVO_RISCO", "modelo_risco_meta.joblib")
ARQUIVO_KMEANS = os.environ.get("ARQUIVO_KMEANS", "modelo_perfil_kmeans.joblib")
# Origens liberadas no CORS (ex.: https://colheai.vercel.app). "*" libera todas.
CORS_ORIGENS = os.environ.get("CORS_ORIGENS", "*")

# Faixas de risco. O modelo acerta cerca de 3 em 4 municipios (F1 0,77), entao
# a API devolve faixas, e nao um "vai/nao vai cumprir" seco.
LIMITE_MEDIO = float(os.environ.get("LIMITE_RISCO_MEDIO", 0.40))
LIMITE_ALTO = float(os.environ.get("LIMITE_RISCO_ALTO", 0.70))

MENSAGENS_RISCO = {
    "BAIXO": "Historico e perfil compativeis com o cumprimento da meta. Manter o "
             "acompanhamento no painel e planejar a subida para 45%.",
    "MEDIO": "Risco moderado. Mapear a oferta local da agricultura familiar e "
             "alinhar a proxima chamada publica ao calendario de safra.",
    "ALTO": "Risco alto de nao cumprir nem a meta antiga de 30%; a meta de 45% "
            "exige acao imediata: mapa de oferta regional, chamada publica "
            "antecipada e apoio aos agricultores no projeto de venda.",
}
ABORDAGEM_PERFIL = {
    "URBANO_DENSO": "Cliente prioritario: muito repasse e pouca oferta local. "
                    "Precisa do mapa de safra regional e de chamadas alinhadas a sazonalidade.",
    "INTERIOR_COMPRA_POUCO_DA_AF": "Cliente de alto risco: ha agricultor por perto, mas a "
                                   "compra nao acontece. Alerta de risco e apoio a chamada publica.",
    "INTERIOR_JA_COMPRA_DA_AF": "Cliente de manutencao: ja compra perto ou acima de 30% e "
                                "precisa chegar a 45%. Painel da meta.",
    "RURAL_COM_MUITA_AF": "Fonte de oferta: seus agricultores podem abastecer os grandes "
                          "centros da regiao.",
}

AVISO_ROTULO = ("O modelo foi treinado com dados reais do FNDE (2011-2019) para prever se a "
                "media de compra da AF fica abaixo de 30%, a meta da epoca. A meta de 45% vale "
                "desde 2026 (Lei 15.226/2025), entao o risco calculado e um PISO.")

# ==============================================================================
# CARGA DOS MODELOS
# ==============================================================================
app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False
if TEM_CORS:
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGENS}})


def carregar(arquivo):
    """Carrega um pacote .joblib. Devolve (pacote, erro) sem derrubar a API."""
    caminho = os.path.join(PASTA_MODELOS, arquivo)
    try:
        return joblib.load(caminho), None
    except Exception as erro:            # arquivo ausente, versao incompativel etc.
        return None, "{}: {}".format(arquivo, erro)


RISCO, ERRO_RISCO = carregar(ARQUIVO_RISCO)
KMEANS, ERRO_KMEANS = carregar(ARQUIVO_KMEANS)
INICIADA_EM = datetime.now(timezone.utc).isoformat(timespec="seconds")

# Campos de entrada em linguagem de negocio (o cliente nao manda log nem 0-1)
CAMPOS_RISCO = {
    "qt_populacao": "populacao do municipio (Censo 2022)",
    "vl_densidade_hab_km2": "densidade demografica (hab/km2)",
    "pc_populacao_rural": "% da populacao na zona rural (0 a 100)",
    "vl_pib_per_capita": "PIB per capita em reais",
    "estab_af_por_mil_hab": "estabelecimentos da agricultura familiar por mil habitantes",
    "vl_transferido_medio": "repasse anual medio do FNDE para o PNAE, em reais (ultimos 3 anos)",
    "pc_af_media_3anos": "% medio do repasse comprado da agricultura familiar (ultimos 3 anos)",
    "qt_anos_sem_compra_af_3anos": "anos sem nenhuma compra da AF (ultimos 3 anos, 0 a 3)",
    "qt_anos_cumpriu_30_6anos": "anos em que comprou 30% ou mais (ultimos 6 anos, 0 a 6)",
    "nm_regiao_intermediaria": "regiao geografica intermediaria do IBGE (ex.: SAO_PAULO, CAMPINAS)",
}
CAMPOS_PERFIL = {
    "qt_populacao": "populacao do municipio (Censo 2022)",
    "vl_densidade_hab_km2": "densidade demografica (hab/km2)",
    "pc_populacao_rural": "% da populacao na zona rural (0 a 100)",
    "vl_pib_per_capita": "PIB per capita em reais",
    "estab_af_por_mil_hab": "estabelecimentos da agricultura familiar por mil habitantes",
    "pc_af_media_3anos": "% medio do repasse comprado da agricultura familiar (ultimos 3 anos)",
}


# ==============================================================================
# PRE-PROCESSAMENTO (o mesmo do notebook, com os parametros do .joblib)
# ==============================================================================
def padronizar_texto(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", "_", re.sub(r"[^A-Za-z0-9 ]", " ", texto.upper()).strip())


def numero(dados, campo, avisos, minimo_log=None):
    """Le um campo numerico. Ausente ou invalido -> None (vira mediana) com aviso."""
    valor = dados.get(campo)
    try:
        valor = float(valor)
        if math.isnan(valor) or math.isinf(valor):
            raise ValueError
    except (TypeError, ValueError):
        avisos.append("Campo '{}' ausente ou invalido: usada a mediana do treino.".format(campo))
        return None
    if minimo_log is not None and valor <= 0:
        avisos.append("Campo '{}' precisa ser maior que zero: usado {}.".format(campo, minimo_log))
        valor = minimo_log
    return valor


def log10(valor):
    return None if valor is None else math.log10(valor)


def minmax(valor, p):
    if p["max"] <= p["min"]:
        return 0.0
    return (valor - p["min"]) / (p["max"] - p["min"])


def preparar_risco(dados):
    avisos = []
    bruto = {
        "log_populacao": log10(numero(dados, "qt_populacao", avisos, 1)),
        "log_densidade": log10(numero(dados, "vl_densidade_hab_km2", avisos, 0.01)),
        "pc_populacao_rural": numero(dados, "pc_populacao_rural", avisos),
        "log_pib_per_capita": log10(numero(dados, "vl_pib_per_capita", avisos, 1)),
        "estab_af_por_mil_hab": numero(dados, "estab_af_por_mil_hab", avisos),
        "log_transferido_medio": log10(numero(dados, "vl_transferido_medio", avisos, 1)),
        "pc_af_media_3anos": numero(dados, "pc_af_media_3anos", avisos),
        "qt_anos_sem_compra_af_3anos": numero(dados, "qt_anos_sem_compra_af_3anos", avisos),
        "qt_anos_cumpriu_30_6anos": numero(dados, "qt_anos_cumpriu_30_6anos", avisos),
    }
    if bruto["pc_af_media_3anos"] is not None:
        bruto["pc_af_media_3anos"] = min(bruto["pc_af_media_3anos"], 100.0)   # mesmo teto do treino
    for coluna, valor in bruto.items():
        if valor is None:
            bruto[coluna] = RISCO["valores_preenchimento"][coluna]

    regiao = padronizar_texto(dados.get("nm_regiao_intermediaria", ""))
    if regiao not in RISCO["regioes"]:
        avisos.append("Regiao '{}' desconhecida: nenhuma coluna de regiao marcada. "
                      "Valores aceitos: {}.".format(regiao, ", ".join(RISCO["regioes"])))
    for r in RISCO["regioes"]:
        bruto["regiao_" + r] = 1.0 if r == regiao else 0.0

    linha = {c: minmax(bruto[c], RISCO["parametros_minmax"][c]) for c in RISCO["colunas"]}
    return pd.DataFrame([linha], columns=RISCO["colunas"]), avisos


def preparar_perfil(dados):
    avisos = []
    bruto = {
        "log_populacao": log10(numero(dados, "qt_populacao", avisos, 1)),
        "log_densidade": log10(numero(dados, "vl_densidade_hab_km2", avisos, 0.01)),
        "pc_populacao_rural": numero(dados, "pc_populacao_rural", avisos),
        "log_pib_per_capita": log10(numero(dados, "vl_pib_per_capita", avisos, 1)),
        "estab_af_por_mil_hab": numero(dados, "estab_af_por_mil_hab", avisos),
        "pc_af_media_2017_2019": numero(dados, "pc_af_media_3anos", avisos),
    }
    for coluna, valor in bruto.items():
        if valor is None:
            bruto[coluna] = KMEANS["valores_preenchimento"][coluna]
    bruto["estab_af_por_mil_hab"] = min(bruto["estab_af_por_mil_hab"], KMEANS["teto_estab_af_por_mil_hab"])
    linha = {c: minmax(bruto[c], KMEANS["parametros_minmax"][c]) for c in KMEANS["colunas"]}
    return pd.DataFrame([linha], columns=KMEANS["colunas"]), avisos


def calcular_risco(dados):
    X, avisos = preparar_risco(dados)
    prob = float(RISCO["modelo"].predict_proba(X)[0, 1])
    faixa = "ALTO" if prob >= LIMITE_ALTO else ("MEDIO" if prob >= LIMITE_MEDIO else "BAIXO")
    return {"probabilidade_abaixo_da_meta": round(prob, 4), "faixa_risco": faixa,
            "mensagem": MENSAGENS_RISCO[faixa], "aviso_rotulo": AVISO_ROTULO,
            "modelo": "LogisticRegression", "versao_modelo": RISCO["versao"], "avisos": avisos}


def calcular_perfil(dados):
    X, avisos = preparar_perfil(dados)
    grupo = int(KMEANS["modelo"].predict(X)[0])
    nome = KMEANS["nomes_grupos"][grupo]
    return {"grupo": grupo, "perfil": nome, "abordagem_colheai": ABORDAGEM_PERFIL[nome],
            "modelo": "KMeans (k = {})".format(KMEANS["metricas"]["k"]),
            "versao_modelo": KMEANS["versao"], "avisos": avisos}


# ==============================================================================
# ROTAS
# ==============================================================================
def corpo_json():
    dados = request.get_json(silent=True)
    return dados if isinstance(dados, dict) else None


def indisponivel(erro):
    return jsonify({"erro": "Modelo indisponivel.", "detalhe": erro}), 503


@app.get("/")
def health():
    return jsonify({
        "servico": "ColheAi - API de IA (Since Young, GS 2026/2)",
        "status": "ok",
        "modelo_risco_carregado": RISCO is not None,
        "modelo_perfil_carregado": KMEANS is not None,
        "iniciada_em": INICIADA_EM,
        "rotas": ["GET /api/status", "POST /api/prever-risco", "POST /api/perfil-municipio",
                  "GET /api/municipio/<cd_ibge>"],
    })


@app.get("/api/status")
def status():
    if RISCO is None or KMEANS is None:
        return indisponivel([e for e in (ERRO_RISCO, ERRO_KMEANS) if e])
    return jsonify({
        "equipe": RISCO["equipe"],
        "fontes_dos_dados": RISCO["fontes"],
        "versao_sklearn_treino": RISCO["versao_sklearn"],
        "modelo_risco": {
            "tipo": RISCO["tipo"], "versao": RISCO["versao"], "rotulo": RISCO["rotulo"],
            "metricas_teste": RISCO["metricas"],
            "faixas": {"BAIXO": "< {}".format(LIMITE_MEDIO),
                       "MEDIO": "{} a {}".format(LIMITE_MEDIO, LIMITE_ALTO),
                       "ALTO": ">= {}".format(LIMITE_ALTO)},
            "campos_aceitos": CAMPOS_RISCO, "regioes_aceitas": RISCO["regioes"],
        },
        "modelo_perfil": {
            "tipo": KMEANS["tipo"], "versao": KMEANS["versao"], "metricas": KMEANS["metricas"],
            "perfis": ABORDAGEM_PERFIL, "campos_aceitos": CAMPOS_PERFIL,
        },
        "municipios_disponiveis": len(RISCO["municipios"]),
    })


@app.post("/api/prever-risco")
def prever_risco():
    if RISCO is None:
        return indisponivel(ERRO_RISCO)
    dados = corpo_json()
    if dados is None:
        return jsonify({"erro": "Envie um objeto JSON com os campos de /api/status."}), 400
    try:
        return jsonify(calcular_risco(dados))
    except Exception as erro:
        return jsonify({"erro": "Falha ao calcular o risco.", "detalhe": str(erro)}), 500


@app.post("/api/perfil-municipio")
def perfil_municipio():
    if KMEANS is None:
        return indisponivel(ERRO_KMEANS)
    dados = corpo_json()
    if dados is None:
        return jsonify({"erro": "Envie um objeto JSON com os campos de /api/status."}), 400
    try:
        return jsonify(calcular_perfil(dados))
    except Exception as erro:
        return jsonify({"erro": "Falha ao calcular o perfil.", "detalhe": str(erro)}), 500


@app.get("/api/municipio/<int:cd_ibge>")
def municipio(cd_ibge):
    if RISCO is None or KMEANS is None:
        return indisponivel([e for e in (ERRO_RISCO, ERRO_KMEANS) if e])
    linha = RISCO["municipios"].get(cd_ibge)
    if linha is None:
        return jsonify({"erro": "Municipio {} nao encontrado. A base cobre os 645 municipios "
                                "de SP (codigo IBGE com 7 digitos, comecando por 35).".format(cd_ibge)}), 404
    # Converte a linha guardada (variaveis 2017-2019) para os campos de negocio
    entrada = {
        "qt_populacao": 10 ** linha["log_populacao"],
        "vl_densidade_hab_km2": 10 ** linha["log_densidade"],
        "pc_populacao_rural": linha["pc_populacao_rural"],
        "vl_pib_per_capita": 10 ** linha["log_pib_per_capita"],
        "estab_af_por_mil_hab": linha["estab_af_por_mil_hab"],
        "vl_transferido_medio": 10 ** linha["log_transferido_medio"],
        "pc_af_media_3anos": linha["pc_af_media_3anos"],
        "qt_anos_sem_compra_af_3anos": linha["qt_anos_sem_compra_af_3anos"],
        "qt_anos_cumpriu_30_6anos": linha["qt_anos_cumpriu_30_6anos"],
        "nm_regiao_intermediaria": linha["nm_regiao_intermediaria"],
    }
    return jsonify({
        "cd_ibge_municipio": cd_ibge,
        "nm_municipio": linha["nm_municipio"],
        "nm_regiao_intermediaria": linha["nm_regiao_intermediaria"],
        "historico_usado": "FNDE 2014-2019 (ultimos anos consolidados antes da pandemia)",
        "dados_entrada": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in entrada.items()},
        "risco_meta": calcular_risco(entrada),
        "perfil": calcular_perfil(entrada),
    })


@app.errorhandler(404)
def nao_encontrado(_):
    return jsonify({"erro": "Rota inexistente. Veja GET / para a lista de rotas."}), 404


@app.errorhandler(405)
def metodo_errado(_):
    return jsonify({"erro": "Metodo HTTP nao permitido nesta rota."}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
