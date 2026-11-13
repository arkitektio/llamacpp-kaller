# llamacpp-kaller

[kaller](https://github.com/jhnnsrs/kaller), the Arkitekt room assistant that runs
actions, driven by a local llama.cpp model instead of an alpaka one.

Asked "segment this" in a room, it finds an installed segmenter, calls it as a child of
the turn that the user approves in the room (unless the turn runs in auto-mode), and
answers with what came back attached. It reads mikro, rekuest, alpaka and kabinet to
understand the data, reads agent state, installs missing apps from the store, remembers
the user and can ask them questions. All of that is kaller's; this app supplies the model.

## How it differs from kaller

kaller's `assign` takes an alpaka `LLMModel` and reaches it through alpaka's tunnel. This
app owns a `llama-server` for a GGUF on the node and hands kaller an OpenAI client pointed
at it. Nothing has to reach the node and no provider has to be registered: the app answers
over the connection it already has. kaller is a dependency, pinned to a commit in
`pyproject.toml`, not a copy.

Two adjustments sit between kaller and the local server (`localkaller/client.py`): a system
message after the first is sent as a user instruction, since some local chat templates
refuse it, and every completion gets a token limit (`LLAMA_MAX_TOKENS`, default 8192) so a
reasoning model cannot think forever.

DeepSeek V4.1 Flash was checked against kaller's pattern on this node: it streamed a
`find_actions` call, then `call_action` with an approval prompt, then a summary after the
"stop calling tools" instruction.

## Setups

A setup is a model and how to serve it: the GGUF or Hugging Face repo, the context,
the `llama-server` flags. They live together in `setups.yaml` beside `app.py`, by name,
with one marked as the default; JSON works too, since it is valid YAML. Start a named one
with `./run.sh qwen` or `LLAMA_PROFILE=qwen`. Single environment variables, or lines in
`.env`, still override any value of the setup. `setups.example.yaml` shows the format and
`localkaller/setups.py` lists every key.

## Running it

```sh
uv sync
uv run arkitekt run dev        # the default setup of setups.yaml
./run.sh qwen                  # or a named one
```

Only one model fits the GPU. Stop the `arkitekt-llamacpp` app first, or set
`LLAMA_ATTACH_URL=http://127.0.0.1:8080` to share its server; starting a second server on a
port that already answers is refused with that advice. The `LLAMA_*` variables are
documented in `localkaller/settings.py`.

In Orkestrator, choose `assign` of this app as the replyer of a room.

## Deploying on another node

What a node needs depends on how the app runs there.

**As an Arkitekt plugin (container).** The flavour builds on llama.cpp's `server-cuda`
image, which exists for amd64 and arm64 and brings `llama-server` and the CUDA runtime;
uv installs Python 3.12 and the dependencies, kaller included, inside the image. The host
needs only Docker with the NVIDIA Container Toolkit and an NVIDIA GPU whose driver runs
CUDA 12.8; the flavour declares a required CUDA selector so it is only placed on such
nodes. Give it a model by mounting a GGUF and setting `LLAMA_MODEL`, or set `LLAMA_HF`
and it downloads one into the `/models` volume on first start.

**Directly, with `arkitekt run`.** The node needs uv (which brings Python 3.12), git, and a
`llama-server` built for its GPU, on `PATH` or named by `LLAMA_SERVER_BIN`. Copy
`setups.example.yaml` to `setups.yaml` and describe the node's models there; its `qwen`
setup downloads its model by itself.

Not covered: nodes without an NVIDIA GPU (there is no CPU or ROCm flavour), and models
larger than the node's GPU and system memory together. DeepSeek V4.1 Flash needs about
430 GB, Qwen3.8-Flash-Next about 115 GB. The base image tag follows llama.cpp's master;
pin it by digest when a build must be reproducible. This lab node's paths live only in its
`setups.yaml`, which is not in git.

## Testing

```sh
uv run pytest
```

The suite runs against `tests/stub_llama_server.py`: the history, the client adjustments,
and a whole turn from room message to streamed answer. kaller's own behaviour is tested in
kaller.

## Layout

- `app.py`: the Arkitekt app: the lifecycle hook, `assign` and `model_info`.
- `localkaller/turn.py`: one kaller turn for one room message.
- `localkaller/client.py`: the OpenAI client kaller gets, with the two adjustments.
- `localkaller/setups.py`: named setups from `setups.yaml`.
- `localkaller/settings.py`, `localkaller/server.py`: the `LLAMA_*` configuration and the server's lifecycle.
