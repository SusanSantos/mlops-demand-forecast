# MLOps — Previsão de Demanda + Monitoramento de Drift

Ciclo completo de ML em produção em miniatura: **treino rastreado (MLflow) → comparação com baseline → monitoramento de drift (PSI + KS + erro) → decisão automática de retreino**.

> **EN summary:** Demand forecasting with GradientBoosting, MLflow experiment tracking, a naive seasonal baseline, temporal train/test split, and a drift-monitoring module (PSI + KS test + error degradation, with seasonal-matched reference and corroboration rule) that recommends retraining.

## Fluxo
```
make train          → treina, loga params/métricas/artefato no MLflow, salva reference.csv e metrics.json
make monitor        → janela "saudável" de produção  → "Modelo saudável"
make monitor-drift  → janela com mudança de regime   → "RETREINAR"
```

## Resultados reais (holdout temporal de 20%)
| Modelo | MAPE |
|---|---|
| Baseline (mesmo dia da semana anterior) | 10,8% |
| GradientBoosting | **5,6%** |

Monitoramento: janela saudável MAPE 5,1% (1 alerta isolado de PSI, ignorado) → *saudável*; janela com drift MAPE 17,6% e 5 features em ALERT → *retreinar*.

## Decisões que valem destacar
- **Split temporal**, nunca aleatório (evita vazamento do futuro).
- **Baseline obrigatório**: o modelo só vale se bater o ingênuo.
- **Referência pareada por sazonalidade**: comparar o ano inteiro com o fim do ano gera falso drift; a referência é filtrada pelo mesmo período do calendário.
- **Regra de corroboração**: um único PSI em ALERT não dispara retreino (PSI é hipersensível em features de baixa variância, como médias móveis); exige ≥ 2 features ou erro > 1,5× o de holdout.
- MAPE de referência vem do **holdout**, não do treino (in-sample seria otimista e geraria alarmes falsos).

## Como rodar
```bash
make install && make train && make monitor && make monitor-drift
make ui     # interface do MLflow
make test && make lint
```

## Limitações
- Dados sintéticos; PSI/KS detectam drift de entrada, não *concept drift* sem rótulos — por isso o monitor de erro quando o real chega.
- Sem registry/promoção automática de modelo; é o próximo passo (MLflow Model Registry + CD).

## Bullets para currículo
- Implementei ciclo MLOps de previsão de demanda com MLflow, split temporal, baseline e monitoramento de drift (PSI, KS, degradação de erro) com decisão automatizada de retreino; modelo reduziu o MAPE de 10,8% (baseline) para 5,6%.
