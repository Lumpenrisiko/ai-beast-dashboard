# 🦖 AI Beast Dashboard

Real-time GPU/LLM monitoring dashboard for LM Studio with live system metrics, token tracking, and MTP draft statistics.

## Features

- **GPU Monitoring**: Utilization, temperature, VRAM, power consumption (per GPU)
- **System Metrics**: CPU temp/usage, RAM usage, disk I/O
- **LM Studio Integration**:
  - Live token generation speed (tok/s) from logs
  - Prompt processing speed (p/s) from logs
  - MTP draft acceptance rate & stats
  - Real-time prompt progress bar
  - Queue length tracking
  - Input/Output token counters (total + session)
- **Network Charts**: Live sparkline charts for GPU, CPU, RAM, power
- **Single HTML UI**: No build step, no dependencies

## Prerequisites

- Python 3.10+
- NVIDIA GPU with `nvidia-smi`
- LM Studio running with Ollama-compatible API (`http://localhost:11434`)
- Optional: [LACT](https://github.com/UnaiEtxebarria/lact) for hotspot/VRAM temps

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-beast-dashboard.git
cd ai-beast-dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure (optional)
cp .env.example .env
# Edit .env with your settings

# Run
python3 main.py
```

Open `http://localhost:8083` in your browser.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `8083` | Dashboard port |
| `LM_STUDIO_URL` | `http://localhost:11434` | LM Studio API URL |
| `LM_STUDIO_LOG_DIR` | `` | Path to LM Studio logs (for real-time metrics) |
| `LACT_ENABLED` | `true` | Enable LACT for hotspot/VRAM temps |
| `STATS_INTERVAL` | `2` | SSE update interval (seconds) |

### LM Studio Log Directory

For real-time token speed metrics, set `LM_STUDIO_LOG_DIR` to your LM Studio log directory:

```bash
# Linux (default LM Studio location)
export LM_STUDIO_LOG_DIR="$HOME/.lmstudio/server-logs/$(date +%Y-%m)"

# The dashboard auto-detects the latest log file
```

## Docker

```bash
# Build
docker build -t ai-beast-dashboard .

# Run
docker run -d \
  --name ai-beast-dashboard \
  -p 8083:8083 \
  -e LM_STUDIO_URL=http://host.docker.internal:11434 \
  ai-beast-dashboard
```

## Systemd Service

```bash
# Copy template and edit paths
cp gpu-dashboard.service.template ~/.config/systemd/user/gpu-dashboard.service
# Edit the service file with your paths

# Enable and start
systemctl --user enable --now gpu-dashboard.service
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | All system stats as JSON |
| `/sse` | GET | Server-Sent Events stream |
| `/v1/chat/completions` | POST | Proxy to LM Studio API |

## Project Structure

```
gpu-dashboard/
├── main.py              # FastAPI backend
├── static/
│   └── index.html       # Frontend UI
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
├── Dockerfile           # Docker build
├── gpu-dashboard.service.template  # Systemd template
├── .gitignore
└── README.md
```

## License

MIT
