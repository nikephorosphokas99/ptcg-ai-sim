# The simulator as a Hugging Face Space (Docker SDK).
#
# Two things make this more than "pip install and run":
#
#  * the engine is C++ and no .so is kept in git, so the image *builds* the
#    engine (build_engine.sh) — the same one command a local checkout runs;
#  * a game is a process, not a request handler. The server spawns one
#    game_worker.py per player and each of those spawns the agent bundle, so the
#    container needs to be able to fork children and hold their memory. That is
#    why PTCG_MAX_SESSIONS exists, and why it defaults low here.
#
# Build stage compiles the engine; the runtime stage carries no compiler.

FROM python:3.12-slim-trixie AS engine

RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY ptcg_engine/src ./ptcg_engine/src
COPY build_engine.sh ./
# The script also installs the .so into every agent bundle's cg/; there are no
# bundles in this stage, so it just writes ptcg_engine/libcg.so.
RUN chmod +x build_engine.sh && ./build_engine.sh


FROM python:3.12-slim-trixie

# Spaces run the container as uid 1000 with $HOME=/home/user. Writing bytecode
# into a read-only-ish tree buys nothing here and only makes the image dirty.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user deploy/requirements.txt ./
# CPU wheels only: a Space has no GPU, and the CUDA build would add ~2.5 GB.
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# The simulator, then the engine binary built in the first stage. Card art is
# copied last: it is 683 MB and never changes, so it stays a cached layer while
# the code above it is edited.
COPY --chown=user . ./
COPY --from=engine --chown=user /build/ptcg_engine/libcg.so ./ptcg_engine/libcg.so
# Every agent bundle runs in its own interpreter and loads its own copy.
RUN for cg in agents/*/cg; do [ -d "$cg" ] && cp ptcg_engine/libcg.so "$cg/"; done; true

# How many games at once. Each costs a worker process plus a torch process, so
# the ceiling is memory: ~1 GB per game, on a 16 GB free CPU Space.
ENV PTCG_MAX_SESSIONS=6 \
    PTCG_IDLE_TIMEOUT=900

EXPOSE 7860
CMD ["python", "server.py"]
