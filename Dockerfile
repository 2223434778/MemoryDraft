# CPU transport demo image. Real GPU deployment uses a separately prepared profile.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY memorydraft ./memorydraft
RUN python -m pip install --no-cache-dir ".[serve,client]"
COPY configs/runtime-demo.json ./configs/runtime-demo.json
COPY examples ./examples
EXPOSE 8000
CMD ["memorydraft", "serve", "--config", "configs/runtime-demo.json", "--host", "0.0.0.0", "--port", "8000"]
