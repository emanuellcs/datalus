# DATALUS: Arquitetura de Difusão Tabular para Utilidade e Segurança Local

> **Português** | [Read the international documentation in English](./README.md)

**Submissão Oficial:** 32º Prêmio Jovem Cientista (2026)  
**Tema:** Inteligência Artificial para o Bem Comum  
**Linha de Pesquisa:** Inteligência Artificial & Tecnologia  

O DATALUS é um *framework* agnóstico de Inteligência Artificial Generativa construído para resolver o paradoxo entre a **Lei Geral de Proteção de Dados (LGPD)** e a necessidade de inovação aberta no Estado brasileiro. 

A arquitetura ingere matrizes governamentais sensíveis (como registros de saúde pública ou educação), aprende a distribuição de probabilidade conjunta subjacente através de Modelos de Difusão Probabilística e gera microdados sintéticos. O resultado é um conjunto de dados estatisticamente perfeito para o treinamento de modelos de *Machine Learning*, mas matematicamente desprovido de qualquer cidadão real, eliminando riscos de reidentificação.

## O Bem Comum e a Aplicação Prática

Projetado sob a realidade da infraestrutura estatal, o DATALUS não exige *clusters* milionários para operar na ponta. O modelo mestre é treinado em nuvem, mas os artefatos de inferência são **quantizados para INT8 via ONNX**. Isso permite que qualquer auditor ou gestor público rode simulações contrafactuais e gere dados sintéticos localmente, na CPU de um computador convencional.

**Prova de Conceito (PoC):** A validação empírica deste repositório foi conduzida sobre os microdados do **DATASUS** (Sistema de Informações Hospitalares), demonstrando a capacidade do *framework* em anonimizar informações clínicas de alta dimensionalidade (como variáveis demográficas cruzadas com códigos CID-10 de alta cardinalidade).

## Engenharia e Componentes

1. **Ingestão Zero-Shot (Camada 1):** Utiliza processamento preguiçoso (*lazy evaluation*) com a biblioteca `Polars` e arquivos Apache Parquet. Permite processar bases governamentais massivas superando gargalos de memória RAM.
2. **Núcleo de Difusão (Camada 3):** Supera as tradicionais Redes Adversárias Generativas (GANs) ao mitigar o colapso de modo. Utiliza reversão de ruído markoviana adaptada para dados heterogêneos (*TabDDPM*).
3. **Orquestrador Autônomo de Auditoria (Camada 5):** O sistema atesta sua própria segurança jurídica e matemática. Para cada base gerada, emite laudos automáticos de:
   * **Privacidade:** Cálculo da Distância Euclidiana para o Registro Mais Próximo (DCR), atestando a impossibilidade de ataques de inferência.
   * **Utilidade (MLE):** Treinamento automático de modelos substitutos (*CatBoost*) na base sintética e validação preditiva ROC-AUC contra dados reais retidos.

## Navegação do Repositório

O ecossistema do projeto divide-se estritamente entre a lógica algorítmica e os artefatos de dados:

* `src/`: O coração matemático do *framework*, escrito em PyTorch, agnóstico a qualquer domínio.
* `deploy/`: Microsserviços encapsulados, incluindo a API em *FastAPI* e os *scripts* de quantização de pesos.
* **Model Registry (Hugging Face):** Os pesos quantizados dos modelos treinados (como o modelo DATASUS) estão hospedados externamente.
* **Data Hub (Kaggle):** Acesse os dados sintéticos finais e os *notebooks* de validação em nosso perfil.

## Conformidade Legal
Este projeto implementa metodologias de anonimização que atendem aos requisitos do Artigo 12 da Lei nº 13.709/2018 (LGPD), assegurando o uso ético da Inteligência Artificial em ambientes de missão crítica.

## Licença
Este projeto está licenciado sob a Licença Apache 2.0.