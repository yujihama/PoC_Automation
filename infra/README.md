# Infra

`docker-compose.langfuse.yml` は、PoC検証用の低スケールLangfuse self-hostテンプレートです。

```bash
cd infra
cp langfuse.env.example .env.langfuse
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml up -d
```

本番または大量実験では、公式Langfuseリポジトリの最新 `docker-compose.yml` またはKubernetes/Helm/Terraform構成に同期してください。
