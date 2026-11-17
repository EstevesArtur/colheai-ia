# -*- coding: utf-8 -*-
"""
================================================================================
COLHEAI - COLETA DOS DADOS PUBLICOS (FNDE + IBGE) PARA OS MODELOS DE IA
Global Solution 2026/2 - CitizenTech - Artificial Intelligence & Chatbot
Since Young - Turma 1TDSPV-2026 - FIAP
--------------------------------------------------------------------------------
Artur Bianchini Esteves                RM569450   REPRESENTANTE
Joao Vitor Oliveira Fontenele Martins  RM570783
--------------------------------------------------------------------------------
O QUE ESTE SCRIPT FAZ

Baixa dados 100% PUBLICOS e REAIS e monta as duas bases brutas da entrega:

  1) FNDE - "PNAE: Dados da Agricultura Familiar" (2011 a 2022).
     O FNDE so publica esses dados num painel Power BI publico, sem arquivo
     para download. O painel, porem, consulta uma API publica do proprio Power
     BI (a mesma que o navegador usa ao abrir o painel). Este script faz a
     mesma consulta, filtrando UF = SP, e recebe: ano, codigo IBGE, Entidade
     Executora, valor transferido pelo FNDE e valor comprado da agricultura
     familiar. Nenhum numero e inventado.

  2) IBGE - API de localidades (regiao intermediaria) e SIDRA:
       tabela 4714 (Censo 2022): populacao, area e densidade
       tabela 9923 (Censo 2022): populacao urbana e rural
       tabela 5938 (PIB dos Municipios 2021)
       tabela 6579 (estimativa de populacao 2021, para o PIB per capita)
       tabela 6778 (Censo Agropecuario 2017): estabelecimentos agropecuarios,
                   total e da agricultura familiar

PROBLEMA DE QUALIDADE ENCONTRADO NA BASE DO FNDE (e corrigido aqui)
  94 linhas de SP vinham com o codigo IBGE de um municipio HOMONIMO de outro
  estado (ex.: "PREF MUN DE TABATINGA" com o codigo 1304062, que e Tabatinga-AM;
  a de SP e 3553609). Quando o codigo nao comeca com 35 (SP), o script
  recupera o codigo certo pelo nome do municipio na lista oficial do IBGE e
  marca a linha com st_codigo_ibge_corrigido = S.

COMO RODAR (opcional: os CSV ja acompanham a entrega)
    pip install pandas
    python coleta_dados_publicos.py
================================================================================
"""

import json
import re
import unicodedata
import urllib.request
from datetime import date

import pandas as pd

DATA_ACESSO = date.today().isoformat()

# ------------------------------------------------------------------------------
# Enderecos das fontes
# ------------------------------------------------------------------------------
PAINEL_FNDE = ("https://www.gov.br/fnde/pt-br/acesso-a-informacao/acoes-e-programas/"
               "programas/pnae/consultas/pnae-dados-da-agricultura-familiar")
PBI_CHAVE = "9622bed6-0d37-4c79-999c-caa8bb19ec1e"   # chave publica do painel
PBI_MODELO = 7024422
PBI_URL = ("https://wabi-brazil-south-api.analysis.windows.net/public/reports/"
           "querydata?synchronous=true")
PBI_TABELA = "Agricultura familiar 2011 a 2022"

IBGE_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"
SIDRA = "https://apisidra.ibge.gov.br/values/t/"


def baixar_json(url, corpo=None, cabecalhos=None):
    """GET ou POST simples que devolve o JSON da resposta."""
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(url, data=dados, headers=cabecalhos or {})
    req.add_header("User-Agent", "Mozilla/5.0 (ColheAi - Since Young - FIAP)")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def padronizar_texto(texto):
    """Regra de CSV do professor: MAIUSCULA, sem acento, _ no lugar de espaco."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^A-Za-z0-9 ]", " ", texto.upper())
    return re.sub(r"\s+", "_", texto.strip())


# ==============================================================================
# 1. FNDE (painel Power BI publico)
# ==============================================================================
def consultar_fnde_sp():
    colunas = ["ANO", "IBGE", "Entidade Executora", "ESFERA", "Valor Transferido",
               "Valor aquisições da agricultura familiar"]
    selecao = [{"Column": {"Expression": {"SourceRef": {"Source": "a"}}, "Property": c},
                "Name": c} for c in colunas]
    filtro_sp = [{"Condition": {"In": {
        "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "a"}},
                                    "Property": "UF"}}],
        "Values": [[{"Literal": {"Value": "'SP'"}}]]}}}]
    consulta = {"Version": 2, "From": [{"Name": "a", "Entity": PBI_TABELA, "Type": 0}],
                "Select": selecao, "Where": filtro_sp}
    corpo = {"version": "1.0.0", "cancelQueries": [], "modelId": PBI_MODELO,
             "queries": [{"Query": {"Commands": [{"SemanticQueryDataShapeCommand": {
                 "Query": consulta,
                 "Binding": {"Primary": {"Groupings": [{"Projections": list(range(len(colunas)))}]},
                             "DataReduction": {"DataVolume": 4,
                                               "Primary": {"Window": {"Count": 30000}}},
                             "Version": 1}}}]}}]}
    resposta = baixar_json(PBI_URL, corpo, {"Content-Type": "application/json",
                                            "X-PowerBI-ResourceKey": PBI_CHAVE})
    return decodificar_dsr(resposta, len(colunas))


def decodificar_dsr(resposta, n_colunas):
    """O Power BI devolve as linhas compactadas (formato DSR):
    - 'R' e uma mascara de bits: coluna repetida da linha anterior;
    - 'Ø' e uma mascara de bits: coluna nula;
    - textos vem como indice de um dicionario (ValueDicts)."""
    ds = resposta["results"][0]["result"]["data"]["dsr"]["DS"][0]
    dicionarios = ds.get("ValueDicts", {})
    esquema, anterior, linhas = None, [None] * n_colunas, []
    for item in ds["PH"][0]["DM0"]:
        esquema = item.get("S", esquema)
        repetidas, nulas = item.get("R", 0), item.get("Ø", 0)
        valores = iter(item.get("C", []))
        atual = []
        for i in range(n_colunas):
            if repetidas & (1 << i):
                valor = anterior[i]
            elif nulas & (1 << i):
                valor = None
            else:
                valor = next(valores)
                chave = esquema[i].get("DN")
                if chave is not None and isinstance(valor, int):
                    valor = dicionarios[chave][valor]
            atual.append(valor)
        anterior = atual
        linhas.append(atual)
    return linhas


# ==============================================================================
# 2. IBGE
# ==============================================================================
def sidra(caminho):
    """Consulta o SIDRA e devolve DataFrame (a primeira linha e o cabecalho)."""
    linhas = baixar_json(SIDRA + caminho)
    return pd.DataFrame(linhas[1:])


def valor_sidra(df, filtro=None):
    if filtro:
        for coluna, valor in filtro.items():
            df = df[df[coluna] == valor]
    # Convencao do IBGE: '-' = zero absoluto; '...' ou 'X' = nao disponivel (vazio)
    serie = pd.to_numeric(df["V"].replace("-", "0"), errors="coerce")
    return pd.Series(serie.values, index=df["D1C"].astype(int))


def montar():
    # ---- municipios e regioes ----
    municipios = baixar_json(IBGE_LOCALIDADES)
    nome_para_codigo = {padronizar_texto(m["nome"]): m["id"] for m in municipios}
    base = pd.DataFrame({
        "cd_ibge_municipio": [m["id"] for m in municipios],
        "nm_municipio": [padronizar_texto(m["nome"]) for m in municipios],
        "nm_regiao_intermediaria": [padronizar_texto(m["regiao-imediata"]["regiao-intermediaria"]["nome"])
                                    for m in municipios],
    }).set_index("cd_ibge_municipio")

    # ---- FNDE ----
    fnde = pd.DataFrame(consultar_fnde_sp(),
                        columns=["ano", "ibge_fnde", "nm_entidade_executora", "esfera",
                                 "vl_transferido", "vl_aquisicao_af"])
    fnde = fnde[fnde["esfera"] == "MUNICIPAL"].copy()        # descarta a rede estadual
    nomes = (fnde["nm_entidade_executora"]
             .str.replace(r"^(PREF MUN DE |PREF MUN DA |MUNICIPIO DE )", "", regex=True)
             .map(padronizar_texto))
    codigo_ok = fnde["ibge_fnde"].astype(str).str.startswith("35")
    fnde["cd_ibge_municipio"] = fnde["ibge_fnde"].astype(int).where(codigo_ok, nomes.map(nome_para_codigo))
    fnde["st_codigo_ibge_corrigido"] = (~codigo_ok).map({True: "S", False: "N"})
    fnde["ano"] = fnde["ano"].astype(int)
    fnde["nm_entidade_executora"] = fnde["nm_entidade_executora"].map(padronizar_texto)
    fnde["pc_aquisicao_af"] = (100 * fnde["vl_aquisicao_af"] / fnde["vl_transferido"]).where(
        fnde["vl_transferido"] > 0)
    fnde = fnde[["ano", "cd_ibge_municipio", "nm_entidade_executora", "vl_transferido",
                 "vl_aquisicao_af", "pc_aquisicao_af", "st_codigo_ibge_corrigido"]]
    fnde = fnde.sort_values(["ano", "cd_ibge_municipio"]).reset_index(drop=True)

    # ---- IBGE: Censo 2022, PIB 2021, Censo Agro 2017 ----
    t4714 = sidra("4714/n6/in%20n3%2035/v/93,6318,614/p/2022")
    base["qt_populacao_2022"] = valor_sidra(t4714, {"D2C": "93"})
    base["vl_area_km2"] = valor_sidra(t4714, {"D2C": "6318"})
    base["vl_densidade_hab_km2"] = valor_sidra(t4714, {"D2C": "614"})

    t9923 = sidra("9923/n6/in%20n3%2035/v/93/p/2022/c1/all")
    base["qt_populacao_rural_2022"] = valor_sidra(t9923, {"D4N": "Rural"})
    base["pc_populacao_rural_2022"] = 100 * base["qt_populacao_rural_2022"] / base["qt_populacao_2022"]

    pib = valor_sidra(sidra("5938/n6/in%20n3%2035/v/37/p/2021"))          # mil reais
    pop21 = valor_sidra(sidra("6579/n6/in%20n3%2035/v/9324/p/2021"))
    base["vl_pib_per_capita_2021"] = 1000 * pib / pop21

    t6778 = sidra("6778/n6/in%20n3%2035/v/183/p/2017/c829/46302,46304")
    base["qt_estab_agro_total_2017"] = valor_sidra(t6778, {"D4C": "46302"})
    base["qt_estab_agro_familiar_2017"] = valor_sidra(t6778, {"D4C": "46304"})
    base["qt_estab_af_por_mil_hab"] = 1000 * base["qt_estab_agro_familiar_2017"] / base["qt_populacao_2022"]

    # ---- FNDE agregado por municipio, em JANELAS DE 3 ANOS ----
    # Percentual acima de 100% conta como 100% (mesmo tratamento do MonitoraSAN/
    # Consea): comprar mais que o repasse nao torna o municipio "mais cumpridor".
    pv_pc = fnde.pivot_table(index="cd_ibge_municipio", columns="ano",
                             values="pc_aquisicao_af").clip(upper=100)
    pv_vt = fnde.pivot_table(index="cd_ibge_municipio", columns="ano", values="vl_transferido")

    def janela(prefixo, anos, anos_meta):
        base[f"vl_transferido_medio_{prefixo}"] = pv_vt[anos].mean(axis=1)
        base[f"pc_aquisicao_af_media_{prefixo}"] = pv_pc[anos].mean(axis=1)
        base[f"qt_anos_sem_compra_af_{prefixo}"] = (pv_pc[anos] == 0).sum(axis=1)
        base[f"qt_anos_cumpriu_30_{anos_meta[0]}_{anos_meta[-1]}"] = (pv_pc[anos_meta] >= 30).sum(axis=1)

    # Janela A (2014-2016): historico usado para TREINAR
    janela("2014_2016", [2014, 2015, 2016], list(range(2011, 2017)))
    # Janela B (2017-2019): ultimos 3 anos consolidados antes da pandemia.
    # E a fonte do ROTULO e tambem o historico que a API usa para pontuar hoje.
    janela("2017_2019", [2017, 2018, 2019], list(range(2014, 2020)))
    # Rotulo: media 2017-2019 abaixo de 30% (meta legal vigente naquele periodo)
    base["st_abaixo_meta_30_2017_2019"] = (base["pc_aquisicao_af_media_2017_2019"] < 30).astype(
        "Int64").where(base["pc_aquisicao_af_media_2017_2019"].notna())
    # 2022: ultimo ano publicado, marcado como PRELIMINAR pelo proprio FNDE.
    # Usado so como teste fora do tempo, nunca no treino.
    base["pc_aquisicao_af_2022_preliminar"] = pv_pc[2022]

    base = base.reset_index()
    for c in base.columns:
        if c.startswith("qt_anos_"):
            base[c] = base[c].fillna(0).astype(int)
    return fnde, base


# Casas decimais fixas por coluna (padrao da Sprint 4)
CASAS = {"vl_area_km2": 3, "vl_densidade_hab_km2": 2, "pc_populacao_rural_2022": 2,
         "vl_pib_per_capita_2021": 2, "qt_estab_af_por_mil_hab": 3,
         "vl_transferido_medio_2014_2016": 2, "pc_aquisicao_af_media_2014_2016": 2,
         "vl_transferido_medio_2017_2019": 2, "pc_aquisicao_af_media_2017_2019": 2,
         "pc_aquisicao_af_2022_preliminar": 2,
         "vl_transferido": 2, "vl_aquisicao_af": 2, "pc_aquisicao_af": 2}
INTEIRAS = ["qt_populacao_2022", "qt_populacao_rural_2022", "qt_estab_agro_total_2017",
            "qt_estab_agro_familiar_2017", "st_abaixo_meta_30_2017_2019", "cd_ibge_municipio"]


def gravar(df, caminho):
    saida = df.copy()
    for coluna, casas in CASAS.items():
        if coluna in saida:
            saida[coluna] = saida[coluna].map(lambda v: "" if pd.isna(v) else f"{v:.{casas}f}")
    for coluna in INTEIRAS:
        if coluna in saida:
            saida[coluna] = saida[coluna].map(lambda v: "" if pd.isna(v) else str(int(v)))
    saida.to_csv(caminho, index=False, encoding="utf-8")   # ausente = celula vazia
    print(f"Gravado: {caminho} ({len(saida)} linhas x {saida.shape[1]} colunas)")


if __name__ == "__main__":
    fnde, base = montar()
    gravar(fnde, "fnde_pnae_agricultura_familiar_sp_2011_2022.csv")
    gravar(base, "municipios_sp_base_bruta.csv")
    print("Data de acesso:", DATA_ACESSO)
