# flood-risk-curitiba

**Mapeamento de áreas de risco de alagamento em Curitiba com Deep Learning**

Segmentação de superfícies impermeáveis a partir de imagens de satélite, integrada à previsão meteorológica operacional, para prever a ocorrência de alagamentos urbanos.

> Trabalho de Conclusão de Curso — Ciência de Dados para Negócios
> FAE Centro Universitário · Curitiba · 2026

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

---

## Sobre o projeto

A impermeabilização do solo urbano converte precipitação em escoamento superficial, sobrecarregando a microdrenagem e produzindo alagamentos recorrentes. Os instrumentos tradicionais de monitoramento são reativos e de atualização lenta.

Este projeto investiga a seguinte questão:

> Como a caracterização da superfície urbana por sensoriamento remoto, combinada à previsão meteorológica operacional, pode antecipar a ocorrência de alagamentos em Curitiba?

A previsão de precipitação já é disponibilizada publicamente por modelos numéricos operacionais. A lacuna não está em prever a chuva, e sim em traduzir a chuva prevista em **onde a água vai acumular**. É esse o problema que o projeto ataca: a previsão meteorológica entra como variável de entrada, e o modelo aprende a relação entre chuva prevista, características físicas do território e ocorrências de alagamento efetivamente registradas.

A entrega prevista é um protótipo funcional que espacializa esse risco e o apresenta de forma acessível a gestores urbanos.

### Objetivos específicos

1. Coletar e pré-processar dados de sensoriamento remoto, previsões meteorológicas operacionais e registros históricos de alagamento para a área de estudo.
2. Desenvolver um modelo de segmentação de superfícies impermeáveis a partir de imagens de satélite.
3. Caracterizar a suscetibilidade física do território, combinando a segmentação a atributos do terreno.
4. Desenvolver um modelo preditivo de ocorrência de alagamento que integre a previsão de precipitação à suscetibilidade mapeada, treinado nos registros históricos.
5. Avaliar o desempenho do modelo frente a baselines de referência.
6. Disponibilizar os resultados em um dashboard interativo.

---

## Escopo

**O que este projeto se propõe a fazer**

- Mapear superfícies impermeáveis em escala intraurbana.
- Consumir previsão meteorológica operacional de fontes públicas como variável de entrada.
- Aprender, a partir de ocorrências historicamente registradas, a relação entre chuva prevista, suscetibilidade do território e alagamento.
- Espacializar o risco resultante e avaliá-lo frente a baselines de referência.

**O que está fora do escopo**

- Produção de previsão meteorológica própria. A previsão de precipitação é insumo do projeto, obtida de modelos numéricos operacionais públicos, e não seu objeto de estudo.
- Modelagem da rede de microdrenagem (galerias, bocas de lobo, capacidade hidráulica), invisível ao sensoriamento remoto.
- Inundação ribeirinha como fenômeno principal — o recorte é **alagamento** por deficiência ou saturação da drenagem urbana.
- Ilhas de calor urbanas: o Sentinel-2 não possui banda no infravermelho termal e, portanto, não deriva Temperatura de Superfície (LST). O tema fica registrado como trabalho futuro.
- Substituição de sistemas oficiais de alerta (Defesa Civil, CEMADEN, SIMEPAR). Trata-se de uma ferramenta de apoio ao planejamento preventivo.

**Definições adotadas**

| Termo | Definição |
|---|---|
| **Alagamento** | Acúmulo temporário de água em vias e áreas urbanas por deficiência ou saturação da microdrenagem. |
| **Inundação** | Transbordamento de curso d'água sobre a planície de inundação. Fora do escopo principal. |
| **Ocorrência** | Registro documentado de alagamento em local e data conhecidos. É a variável-alvo do modelo preditivo. |

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
