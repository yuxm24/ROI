FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
WORKDIR /app
COPY shared ./shared
COPY inference ./inference
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ./shared ./inference
COPY configs ./configs
RUN useradd --uid 10001 --create-home zwam
USER zwam
ENTRYPOINT ["python", "-m", "zwam.server"]
CMD ["--config", "/app/configs/server.local.json"]
