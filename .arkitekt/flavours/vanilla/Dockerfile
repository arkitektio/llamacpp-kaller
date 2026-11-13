# llama.cpp's own server image brings llama-server and the CUDA runtime, for amd64 and
# arm64; the app and its Python come on top. The model is not in the image: mount a GGUF
# and set LLAMA_MODEL, or set LLAMA_HF and llama-server downloads it into /models.
# The tag moves with llama.cpp's master: pin it (by digest) for reproducible builds.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# git: uv fetches the kaller dependency from GitHub. ca-certificates: for that and the downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# /app is the base image's: llama-server and its libraries live there.
WORKDIR /app

# Outside /app, so copying the project below cannot replace it with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# uv brings its own Python 3.12; the base image has none.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
COPY pyproject.toml uv.lock .python-version /app/
RUN uv python install && uv sync --frozen --no-dev --no-install-project

ENV PATH="/opt/venv/bin:$PATH"
ENV LLAMA_SERVER_BIN=/app/llama-server
ENV LLAMA_LOG_DIR=/logs
ENV LLAMA_CACHE=/models
VOLUME /models /logs

# The whole project: the app target may be any module in it, not only app.py.
COPY . /app
RUN uv sync --frozen --no-dev

# The image's own entrypoint is llama-server; the app starts it itself.
ENTRYPOINT []

# The version of this build, when it is not the one the source declares: a build from
# a branch is passed it by `arkitekt plugin release`. Empty for a release proper.
ARG ARKITEKT_APP_VERSION=""
ENV ARKITEKT_APP_VERSION=$ARKITEKT_APP_VERSION
