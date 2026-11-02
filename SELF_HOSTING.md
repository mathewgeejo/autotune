# Self-hosting Voice Studio

The browser is the client: phones, tablets, and computers connect to one server. Audio inference runs on that server, not on each phone. No paid speech API is configured.

## Windows

Run `setup.cmd` and then `launch.cmd` for local access at http://127.0.0.1:7860. For access from your home network:

```powershell
.venv/Scripts/python.exe app.py --no-browser --host 0.0.0.0 --port 7860
```

Visit `http://<server-LAN-IP>:7860` on another device on the same network. Upload works over LAN HTTP. Mobile browsers require HTTPS (or localhost) for microphone permission; use an HTTPS reverse proxy for recording on a remote device. Changing the host does not configure your router or expose a public address automatically.

## Docker Compose

```sh
docker compose up --build -d
docker compose logs -f voice-studio
```

The default binding is localhost. The named volume `voice-studio-data` preserves models, reference conditioning, results, and diagnostics across container replacement. Stopping the container preserves the volume:

```sh
docker compose down
```

Do not add `--volumes` unless you intend to delete the stored data. Recordings and model weights are excluded from the Docker build context.

Copy `.env.example` to `.env` to configure access. Use `VOICE_STUDIO_BIND=0.0.0.0` for LAN access. Set both `VOICE_STUDIO_USERNAME` and `VOICE_STUDIO_PASSWORD` for a private instance shared across your devices. Use a separate app password, not a GitHub password. All devices signed into this instance share its recordings and storage controls.

## Linux without Docker

Use Python 3.10 and install `libsndfile1` and `libgomp1` through your system package manager. Then:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python app.py --no-browser --host 0.0.0.0
```

The dependency lock includes platform markers for Windows and Linux. Linux workers start in their own process group; Cancel terminates the group. The Docker image uses Tini to forward termination and reap child processes. Windows uses process-tree termination without visible helper windows.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOICE_STUDIO_DATA_DIR` | Project `data/` | Writable recordings/cache directory; `/data` in the image. |
| `VOICE_STUDIO_HOST` | `127.0.0.1` | Listen address; `0.0.0.0` inside the image. |
| `PORT` | `7860` | HTTP listen port. |
| `VOICE_STUDIO_THREADS` | `2` | CPU threads for inference. |
| `VOICE_STUDIO_USERNAME`, `VOICE_STUDIO_PASSWORD` | Empty | Optional instance login; configure both together. |
| `VOICE_STUDIO_REQUIRE_AUTH` | Unset | Set to `1` for a public instance; startup fails if login is missing. |
| `HF_HUB_OFFLINE` | Unset | Set to `1` after caching all models to forbid model-network lookups. |

For HTTPS behind a reverse proxy, forward all routes, WebSockets, and streaming queue responses. Allow uploads of up to 150 MB and long-lived generation requests. The app never creates a Gradio public sharing link.

## Models and memory

English uses the existing Chatterbox Turbo ONNX engine. Hindi and Malayalam use a pinned public DhVaani ONNX export, with full-precision 32-step Detailed and 16-step Balanced modes. Indic reference words are transcribed when not supplied. The selected model conditions on a focused reference window (up to ten seconds) even when the uploaded sample is a full minute; the UI shows exactly which section is used.

Full quality audit uses a separate speaker verifier and Whisper large-v3-turbo recognizer. It releases the synthesis engine before loading these models. Allow several GB of RAM and model storage; CPU generation can take minutes. The interface does not promise real-time inference or an indistinguishable clone on every voice.

## Zero-cost cloud constraint

Deployment is the final project gate, not part of setup. No cloud resource or paid plan is provisioned by these files. As checked in November 2026, new Hugging Face compute Spaces require a paid plan even though CPU Basic has no hourly charge. Small free web tiers with around 512 MB RAM are insufficient for the complete inference and quality-audit stack. A free credit is not a guarantee of zero charges after the credit runs out.

Only a verified existing free allocation with sufficient resources, a confirmed grant, or a genuinely free provider configuration may be used. No paid fallback, billing upgrade, or automatic overage is authorized. A temporary tunnel to this PC would provide remote HTTPS access, but would not constitute cloud-hosted inference; it must be described accordingly.

## Current verification limits

This workstation has neither Docker nor WSL installed. The Dockerfile and Compose configuration are prepared, but a real Linux container run remains a required gate. Do not interpret a configuration parse as a successful image build or deployment. `WORK_PLAN.md` tracks the remaining quality and deployment gates.
