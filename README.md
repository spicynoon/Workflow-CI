# Workflow-CI (MLflow Projects + Docker)

- `MLProject/` berisi MLflow Project untuk melatih model Adult Income.
- GitHub Actions (`ci.yml`) akan:
  1) `mlflow run` proyek ini
  2) upload artifacts (`mlruns/**`)
  3) generate Dockerfile dari model MLflow
  4) build & push image ke Docker Hub: `spicynoon/adult-income:latest`
