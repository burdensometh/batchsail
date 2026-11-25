ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG OFFLINE_BUILD=0
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN if [ "$OFFLINE_BUILD" = "1" ]; then \
        python -m pip install --disable-pip-version-check --no-cache-dir --no-deps --no-build-isolation .; \
    else \
        python -m pip install --disable-pip-version-check --no-cache-dir .; \
    fi
RUN mkdir /state && chown 65532:65532 /state
USER 65532:65532
ENTRYPOINT ["batchsail"]
CMD ["serve", "--workspace", "/state"]
