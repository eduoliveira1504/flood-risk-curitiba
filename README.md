# flood-risk-curitiba

**Mapeamento de áreas de risco de alagamento em Curitiba com Deep Learning**

Segmentação de superfícies impermeáveis a partir de imagens de satélite integrada à previsão de precipitação de curto prazo para composição de um índice de risco de alagamento urbano.

> Trabalho de Conclusão de Curso — Ciência de Dados para Negócios
> FAE Centro Universitário · Curitiba · 2026

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

---

## Sobre o projeto

A impermeabilização do solo urbano converte precipitação em escoamento superficial, sobrecarregando a microdrenagem e produzindo alagamentos recorrentes. Os instrumentos tradicionais de monitoramento são reativos e de atualização lenta.

Este projeto investiga a seguinte questão:

> Como a integração entre a segmentação de superfícies impermeáveis por sensoriamento remoto e a previsão de precipitação de curto prazo pode apoiar a identificação de áreas suscetíveis a alagamento em Curitiba?

A entrega prevista é um protótipo funcional que combina a análise espacial da superfície urbana com a modelagem temporal da precipitação, resultando em um índice de risco espacializado e apresentado de forma acessível a gestores urbanos.

### Objetivos específicos

1. Coletar e pré-processar dados de sensoriamento remoto e séries meteorológicas históricas para a área de estudo.
2. Desenvolver um modelo de segmentação de superfícies impermeáveis.
3. Desenvolver um modelo de previsão de precipitação de curto prazo.
4. Compor um índice de risco de alagamento a partir da integração desses resultados.
5. Validar o índice contra registros históricos de alagamento.
6. Disponibilizar os resultados em um dashboard interativo.

---

## Escopo

**O que este projeto se propõe a fazer**

- Mapear superfícies impermeáveis em escala intraurbana.
- Prever precipitação em horizonte curto, sempre comparada a um baseline de referência.
- Combinar ameaça (chuva prevista) e suscetibilidade (características físicas do terreno) em um índice de risco espacializado.
- Validar o índice contra ocorrências históricas registradas.

**O que está fora do escopo**

- Modelagem da rede de microdrenagem (galerias, bocas de lobo, capacidade hidráulica), invisível ao sensoriamento remoto.
- Inundação ribeirinha como fenômeno principal — o recorte é **alagamento** por deficiência ou saturação da drenagem urbana.
- Ilhas de calor urbanas: o Sentinel-2 não possui banda no infravermelho termal e, portanto, não deriva Temperatura de Superfície (LST). O tema fica registrado como trabalho futuro.
- Substituição de sistemas oficiais de alerta (Defesa Civil, CEMADEN, SIMEPAR). Trata-se de uma ferramenta de apoio ao planejamento preventivo.

**Definições adotadas**

| Termo | Definição |
|---|---|
| **Alagamento** | Acúmulo temporário de água em vias e áreas urbanas por deficiência ou saturação da microdrenagem. |
| **Inundação** | Transbordamento de curso d'água sobre a planície de inundação. Fora do escopo principal. |

---

## Equipe

| Integrante |
|---|
| Ana Júlia Luciano Kucharski |
| Eduardo de Oliveira Pereira |
| Henrique Bueno de Carvalho Reis |
| Jhonatan Atanazio |
| Vinicius Antunes do Prado |

**Orientação:** Prof. Msc. Eng. Eunelson José da Silva Júnior
**Orientação na primeira etapa (TCC I):** Prof.ª Dra. Julianna Crippa

**Instituição:** FAE Centro Universitário — Curso de Ciência de Dados para Negócios
