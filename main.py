#!/usr/bin/env python3
"""AI Beast Dashboard — FastAPI Backend (Log-only, no proxy)"""

import asyncio
import json
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import aiohttp
import psutil
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ─── Configuration (from environment variables) ───
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8083"))
DASHBOARD_MODE = os.getenv("DASHBOARD_MODE", "lmstudio").lower()  # "lmstudio" or "ollama"
# Backend URLs (for API checks only, no proxy)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LM_STUDIO_LOG_DIR = os.getenv("LM_STUDIO_LOG_DIR", "")
# Auto-detect current month's log directory if not explicitly set
if not LM_STUDIO_LOG_DIR:
    _default_base = os.path.expanduser("~/.lmstudio/server-logs")
    _current_month = datetime.now().strftime("%Y-%m")
    _auto_dir = os.path.join(_default_base, _current_month)
    if os.path.isdir(_auto_dir):
        LM_STUDIO_LOG_DIR = _auto_dir
    else:
        # Fallback: find the most recent month directory
        if os.path.isdir(_default_base):
            months = sorted([d for d in os.listdir(_default_base) if os.path.isdir(os.path.join(_default_base, d))], reverse=True)
            if months:
                LM_STUDIO_LOG_DIR = os.path.join(_default_base, months[0])
LACT_ENABLED = os.getenv("LACT_ENABLED", "true").lower() == "true"
STATS_INTERVAL = int(os.getenv("STATS_INTERVAL", "2"))
CHART_HISTORY = 60  # seconds of chart data to keep
# Token pricing (EUR per 1M tokens)
COST_INPUT_PER_M = float(os.getenv("COST_INPUT_PER_M", "0.325"))
COST_OUTPUT_PER_M = float(os.getenv("COST_OUTPUT_PER_M", "1.95"))


class LlmLogParser:
    """Parse llama.cpp logs (LM Studio or Ollama) in real-time for accurate metrics.
    
    Supports two modes:
    - LM Studio: Watch log files in LM_STUDIO_LOG_DIR
    - Ollama: Poll journalctl -u ollama.service
    """

    def __init__(self):
        self._current_file = None
        self._position = 0
        self._last_mtime = 0
        # Ollama journalctl cursor (prevents duplicate counting)
        self._ollama_cursor = ""
        self._latest = {
            "prompt_progress": 0,
            "prompt_tokens_per_sec": 0,
            "tokens_per_sec": 0,
            "draft_acceptance": 0,
            "draft_accepted": 0,
            "draft_generated": 0,
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "prompt_time_ms": 0,
            "eval_time_ms": 0,
            "model": "",
            "has_timing": False,
            "last_update": 0,
            # TTL tracking for real-time values
            "tok_s_time": 0,
            "p_s_time": 0,
            # Cumulative token counters (persistent, never reset)
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            # Session token counters (resettable via /api/reset-session-tokens)
            "session_input_tokens": 0,
            "session_output_tokens": 0,
            # Queue tracking
            "queue_length": 0,
            "last_queue_update": 0,
        }
        self._lock = asyncio.Lock()
        self._watch_task = None
        # TTL in seconds - values expire after this
        self._ttl = 5

    async def start_watching(self):
        """Start watching logs."""
        if self._watch_task and not self._watch_task.done():
            return
        self._watch_task = asyncio.ensure_future(self._watch_loop())

    async def stop_watching(self):
        """Stop watching logs."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    def _get_latest_log(self) -> str:
        """Get the path to the latest log file for today (numeric sort)."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            def sort_key(f):
                try:
                    num = int(f.replace(today + '.', '').replace('.log', ''))
                except ValueError:
                    num = 0
                return num
            files = sorted(os.listdir(LM_STUDIO_LOG_DIR), reverse=True, key=sort_key)
            for f in files:
                if f.startswith(today) and f.endswith(".log"):
                    return os.path.join(LM_STUDIO_LOG_DIR, f)
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return None

    async def _watch_loop(self):
        """Continuously read new log lines (LM Studio mode) or poll journalctl (Ollama mode)."""
        while True:
            try:
                await asyncio.sleep(0.3)
                if DASHBOARD_MODE == "ollama":
                    await self._poll_ollama_logs()
                else:
                    await self._watch_lmstudio_logs()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _watch_lmstudio_logs(self):
        """Watch LM Studio log files."""
        latest = self._get_latest_log()
        if not latest:
            return

        try:
            mtime = os.path.getmtime(latest)
            fsize = os.path.getsize(latest)
        except OSError:
            return

        file_changed = (latest != self._current_file or mtime != self._last_mtime)
        if file_changed:
            self._current_file = latest
            self._last_mtime = mtime
            try:
                with open(latest, "r") as f:
                    seek_pos = max(0, fsize - 100000)  # Last 100KB
                    f.seek(seek_pos)
                    content = f.read()
                    lines = content.split("\n")
                    start_idx = 0
                    if seek_pos > 0 and lines and not lines[0].startswith("["):
                        start_idx = 1
                    for line in lines[start_idx:]:
                        if line:
                            self._parse_line(line)
                    self._position = fsize
            except (FileNotFoundError, PermissionError):
                pass
            return

        # Read new lines appended since last check
        try:
            with open(latest, "r") as f:
                f.seek(self._position)
                new = f.read()
                self._position = f.tell()
                for line in new.split("\n"):
                    if line:
                        self._parse_line(line)
        except (FileNotFoundError, PermissionError):
            self._current_file = None
            self._position = 0

    async def _poll_ollama_logs(self):
        """Poll Ollama logs from journalctl using cursor to avoid duplicates."""
        try:
            cmd = ["journalctl", "-u", "ollama.service", "--no-pager", "--output=cat"]
            if self._ollama_cursor:
                cmd.extend(["--cursor", self._ollama_cursor])
            else:
                # First run: only get recent logs (last 30s)
                cmd.extend(["--since", "30 seconds ago"])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            output = stdout.decode()
            if not output:
                return
            # Get the cursor from the last line for next poll
            # journalctl outputs cursor on stderr with --output=json, but we can
            # track by using the last timestamp as a marker instead
            for line in output.split("\n"):
                if line:
                    self._parse_line(line)
            # Update cursor: get the cursor value from the latest journal entry
            try:
                cursor_proc = await asyncio.create_subprocess_exec(
                    "journalctl", "-u", "ollama.service", "--no-pager",
                    "--output=json", "--lines=1",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                cursor_out, _ = await asyncio.wait_for(cursor_proc.communicate(), timeout=2)
                if cursor_out:
                    lines = cursor_out.strip().split(b"\n")
                    if lines:
                        last_entry = json.loads(lines[-1])
                        self._ollama_cursor = last_entry.get("__CURSOR", "")
            except Exception:
                pass
        except Exception:
            pass

    def _parse_line(self, line: str):
        """Parse a single log line for metrics (works for both LM Studio and Ollama)."""
        # Model name from [INFO][model_name] lines (LM Studio)
        m = re.search(r"\[INFO\]\[(\S+)\]", line)
        if m:
            self._latest["model"] = m.group(1)

        # Model name from Ollama load_model lines: load_model: name='qwen3.6_35b_mtp-opti:latest'
        m = re.search(r"load_model:\s*name='([^']+)'", line)
        if m:
            self._latest["model"] = m.group(1)

        # Prompt processing progress: 55.8%
        m = re.search(r"Prompt processing progress:\s*([\d.]+)%", line)
        if m:
            self._latest["prompt_progress"] = float(m.group(1))
            self._latest["last_update"] = time.time()
            return

        # REAL-TIME: prompt processing, n_tokens = 57344, progress = 0.52, t = 49.08 s / 1168.47 tokens per second
        m = re.search(r"prompt processing.*?n_tokens\s*=\s*(\d+).*?progress\s*=\s*([\d.]+).*?t\s*=\s*[\d.]+\s*s\s*/\s*([\d.]+)\s*tokens per second", line)
        if m:
            self._latest["prompt_tokens"] = int(m.group(1))
            self._latest["prompt_progress"] = float(m.group(2)) * 100
            self._latest["prompt_tokens_per_sec"] = float(m.group(3))
            self._latest["p_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            return

        # REAL-TIME: n_decoded = 100, tg = 54.36 t/s (token generation speed)
        m = re.search(r"n_decoded\s*=\s*(\d+),?\s*tg\s*=\s*([\d.]+)\s*t/s", line)
        if m:
            self._latest["n_decoded"] = int(m.group(1))
            self._latest["tokens_per_sec"] = float(m.group(2))
            self._latest["tok_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            return

        # prompt eval time = 80014.57 ms / 86423 tokens (0.93 ms per token, 1080.09 tokens per second)
        m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(.*?,\s*([\d.]+)\s*tokens per second\)", line)
        if m:
            self._latest["prompt_time_ms"] = float(m.group(1))
            self._latest["prompt_tokens"] = int(m.group(2))
            self._latest["prompt_tokens_per_sec"] = float(m.group(3))
            self._latest["p_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            # Track cumulative input tokens (both total and session)
            n_tokens = int(m.group(2))
            self._latest["total_input_tokens"] += n_tokens
            self._latest["session_input_tokens"] += n_tokens
            return

        # eval time = 2209.21 ms / 181 tokens (12.21 ms per token, 81.93 tokens per second)
        m = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(.*?,\s*([\d.]+)\s*tokens per second\)", line)
        if m:
            self._latest["eval_time_ms"] = float(m.group(1))
            self._latest["eval_tokens"] = int(m.group(2))
            self._latest["tokens_per_sec"] = float(m.group(3))
            self._latest["tok_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            # Track cumulative output tokens (both total and session)
            n_tokens = int(m.group(2))
            self._latest["total_output_tokens"] += n_tokens
            self._latest["session_output_tokens"] += n_tokens
            return

        # draft acceptance = 0.82738 (139 accepted / 168 generated)
        m = re.search(r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", line)
        if m:
            self._latest["draft_acceptance"] = float(m.group(1))
            self._latest["draft_accepted"] = int(m.group(2))
            self._latest["draft_generated"] = int(m.group(3))
            self._latest["draft_rate"] = float(m.group(1)) * 100
            self._latest["last_update"] = time.time()

        # Ollama MTP draft stats: statistics        draft-mtp: #calls(b,g,a) = 3 593 593, #gen drafts = 593, #acc drafts = 518, #gen tokens = 2371, #acc tokens = 1638
        m = re.search(r"statistics\s+draft-mtp:.*?#gen drafts\s*=\s*(\d+).*?#acc drafts\s*=\s*(\d+).*?#gen tokens\s*=\s*(\d+).*?#acc tokens\s*=\s*(\d+)", line)
        if m:
            gen_drafts = int(m.group(1))
            acc_drafts = int(m.group(2))
            gen_tokens = int(m.group(3))
            acc_tokens = int(m.group(4))
            self._latest["draft_generated"] = gen_drafts
            self._latest["draft_accepted"] = acc_drafts
            self._latest["draft_tokens_generated"] = gen_tokens
            self._latest["draft_tokens_accepted"] = acc_tokens
            if gen_drafts > 0:
                self._latest["draft_acceptance"] = acc_drafts / gen_drafts
                self._latest["draft_rate"] = (acc_drafts / gen_drafts) * 100
            self._latest["last_update"] = time.time()

        # Queue tracking: "all slots are idle" = queue is empty
        if "all slots are idle" in line:
            self._latest["queue_length"] = 0
            self._latest["last_queue_update"] = time.time()

        # Queue tracking: active tasks (slot launch or processing)
        m = re.search(r"launch_slot_.*task\s+(\d+)\s*\|", line)
        if m:
            self._latest["queue_length"] = max(self._latest["queue_length"], 1)
            self._latest["last_queue_update"] = time.time()

        # Model name from Ollama logs: model name in timing lines
        m = re.search(r"id\s+\d+\s*\|\s*task\s+\d+\s*\|.*?(qwen|gemma|llama|mistral|nemotron)[^\s|]*", line, re.IGNORECASE)
        if m and not self._latest["model"]:
            self._latest["model"] = m.group(0).split("|")[0].strip()

    async def get_latest(self) -> dict:
        """Get the latest parsed metrics with TTL expiration."""
        async with self._lock:
            result = dict(self._latest)
            now = time.time()

            progress = result.get("prompt_progress", 0)

            # Apply TTL: zero out expired values
            if now - result.get("tok_s_time", 0) > self._ttl:
                result["tokens_per_sec"] = 0
            if now - result.get("p_s_time", 0) > self._ttl:
                result["prompt_tokens_per_sec"] = 0

            # Context-based zeroing:
            # - When progress == 100% (generation): p/s should be 0
            # - When progress < 100% (prompt processing): tok/s should be 0
            if progress >= 100:
                result["prompt_tokens_per_sec"] = 0
            if progress > 0 and progress < 100:
                result["tokens_per_sec"] = 0

            # Reset prompt_progress if idle
            if now - result["last_update"] > 10 and not result["has_timing"]:
                result["prompt_progress"] = 0

            # Queue TTL: reset if no update for 15s
            if now - result.get("last_queue_update", 0) > 15:
                result["queue_length"] = 0

            # Remove internal tracking fields from output
            result.pop("tok_s_time", None)
            result.pop("p_s_time", None)
            result.pop("last_queue_update", None)

            return result


# Global log parser instance
log_parser = LlmLogParser()


# ─── System stats collectors ───────────────────────────────────────
async def get_gpu_stats() -> list:
    """Get GPU stats from nvidia-smi + LACT for hotspot/VRAM temps."""
    try:
        result = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,temperature.gpu,"
            "temperature.memory,memory.used,memory.total,power.draw,power.limit,"
            "clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=10)
        lines = stdout.decode().strip().split("\n")
        gpus = []
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 14:
                mem_temp = parts[5].strip()
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_gpu": float(parts[2]),
                    "util_mem": float(parts[3]),
                    "temperature": float(parts[4]),
                    "mem_temperature": float(mem_temp) if mem_temp != "N/A" else None,
                    "mem_used": float(parts[6]),
                    "mem_total": float(parts[7]),
                    "power_draw": float(parts[8]),
                    "power_limit": float(parts[9]),
                    "clock_graphics": float(parts[10]),
                    "clock_memory": float(parts[11]),
                    "clock_max_graphics": float(parts[12]),
                    "clock_max_memory": float(parts[13]),
                    "hotspot_temp": None,
                })

        # Get LACT stats for hotspot/VRAM temps
        lact_stats = await _get_lact_stats()
        for gpu in gpus:
            idx = gpu["index"]
            if idx in lact_stats:
                gpu["hotspot_temp"] = lact_stats[idx].get("hotspot")
                if gpu["mem_temperature"] is None:
                    gpu["mem_temperature"] = lact_stats[idx].get("vram")

        return gpus
    except Exception as e:
        return [{"error": str(e)}]


async def _get_lact_stats() -> dict:
    """Get hotspot and VRAM temps from LACT (optional)."""
    if not LACT_ENABLED:
        return {}
    result = {}
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--list-gpus",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        gpu_count = len([l for l in stdout.decode().strip().split("\n") if l.strip()])
        if gpu_count == 0:
            return {}

        # Build PCI bus mapping
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        nvidia_pci = {}
        for line in stdout.decode().strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    nvidia_pci[parts[1].strip()] = int(parts[0])

        for lact_id in range(1, gpu_count + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "lact", "cli", "-g", str(lact_id), "stats",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                text = stdout.decode()

                pci_match = re.search(r"0000:(\w+):(\d+)\.(\d+)", text)
                if not pci_match:
                    continue
                pci_bus = f"0000:{pci_match.group(1)}:{pci_match.group(2)}.{pci_match.group(3)}"

                nvidia_idx = None
                for pci, idx in nvidia_pci.items():
                    pci_normalized = [p.lower() for p in pci.split(":")[1:]]
                    bus_normalized = [p.lower() for p in pci_bus.split(":")[1:]]
                    if pci_normalized == bus_normalized:
                        nvidia_idx = idx
                        break
                if nvidia_idx is None:
                    continue

                hotspot = vram = None
                for line in text.split("\n"):
                    if "Temperatures:" in line:
                        if "GPU Hotspot:" in line:
                            m = re.search(r"GPU Hotspot:\s*(\d+)°C", line)
                            if m:
                                hotspot = int(m.group(1))
                        if "VRAM:" in line:
                            m = re.search(r"VRAM:\s*(\d+)°C", line)
                            if m:
                                vram = int(m.group(1))
                result[nvidia_idx] = {"hotspot": hotspot, "vram": vram}
            except (asyncio.TimeoutError, Exception):
                continue
    except Exception:
        pass
    return result


async def get_cpu_stats() -> dict:
    """Get CPU stats."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    cpu_count = psutil.cpu_count()

    temp = None
    try:
        result = await asyncio.create_subprocess_exec(
            "sensors", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=5)
        for line in stdout.decode().split("\n"):
            if "Tctl" in line:
                match = re.search(r"([\d.]+)", line.split(":")[-1])
                if match:
                    temp = float(match.group(1))
                    break
    except Exception:
        pass

    return {
        "percent": cpu_percent,
        "freq_current": cpu_freq.current if cpu_freq else None,
        "temperature": temp,
        "cores": cpu_count,
    }


async def get_memory_stats() -> dict:
    """Get RAM stats."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "used_gb": round(mem.used / (1024**3), 1),
        "percent": mem.percent,
    }


async def get_disk_stats() -> dict:
    """Get disk stats."""
    disk = psutil.disk_usage("/home")
    return {
        "total_gb": round(disk.total / (1024**3), 1),
        "used_gb": round(disk.used / (1024**3), 1),
        "percent": disk.percent,
    }


# ─── LLM API helpers (no proxy, direct API calls) ──────────────────
async def check_llm_running() -> bool:
    """Check if the LLM backend is running by querying its API."""
    try:
        url = OLLAMA_URL if DASHBOARD_MODE == "ollama" else LM_STUDIO_URL
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def get_loaded_models() -> list:
    """Get list of models currently loaded in VRAM (actively running)."""
    models = []
    try:
        url = OLLAMA_URL if DASHBOARD_MODE == "ollama" else LM_STUDIO_URL
        async with aiohttp.ClientSession() as session:
            # Ollama /api/ps shows models currently loaded in memory
            # LM Studio: no direct endpoint, infer from /v1/models + VRAM usage
            if DASHBOARD_MODE == "ollama":
                endpoint = f"{url}/api/ps"
            else:
                endpoint = f"{url}/v1/models"
            async with session.get(
                endpoint,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if DASHBOARD_MODE == "ollama":
                        for m in data.get("models", []):
                            models.append(m.get("name", m.get("model", "")))
                    else:
                        # LM Studio doesn't have a "loaded" endpoint
                        # (loaded_models will be populated via log parser)
                        pass
    except Exception:
        pass
    return models


async def get_available_models() -> list:
    """Get list of all available models on disk."""
    models = []
    try:
        url = OLLAMA_URL if DASHBOARD_MODE == "ollama" else LM_STUDIO_URL
        async with aiohttp.ClientSession() as session:
            if DASHBOARD_MODE == "ollama":
                endpoint = f"{url}/api/tags"
            else:
                endpoint = f"{url}/v1/models"
            async with session.get(
                endpoint,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if DASHBOARD_MODE == "ollama":
                        for m in data.get("models", []):
                            models.append(m.get("name", ""))
                    else:
                        for m in data.get("data", []):
                            models.append(m.get("id", ""))
    except Exception:
        pass
    return models


# ─── FastAPI App ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await log_parser.start_watching()
    yield
    _shutdown_flag.set()
    await log_parser.stop_watching()

app = FastAPI(title="GPU/LLM Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/stats")
async def api_stats():
    """Get all system stats as JSON."""
    try:
        return await asyncio.wait_for(_collect_stats(), timeout=15)
    except asyncio.TimeoutError:
        return {"error": "Stats collection timed out"}


@app.post("/api/reset-session-tokens")
async def api_reset_session_tokens():
    """Reset only the session token counter. Total counter persists."""
    async with log_parser._lock:
        log_parser._latest["session_input_tokens"] = 0
        log_parser._latest["session_output_tokens"] = 0
    return {"success": True}


# ─── Global shutdown flag + SSE connection tracking ─────────────────
_shutdown_flag = asyncio.Event()


# ─── SSE Stream ────────────────────────────────────────────────────
@app.get("/sse")
async def sse_stream():
    """Server-Sent Events for live dashboard updates."""
    async def event_stream():
        while not _shutdown_flag.is_set():
            try:
                task = asyncio.create_task(_collect_stats())
                done, pending = await asyncio.wait(
                    [task], timeout=STATS_INTERVAL, return_when=asyncio.FIRST_COMPLETED
                )
                for p in pending:
                    p.cancel()

                if _shutdown_flag.is_set():
                    break

                if task in done:
                    data = task.result()
                    yield f"data: {json.dumps(data)}\n\n"

                # Wait for remaining interval to maintain steady tick rate
                elapsed = time.time() - (getattr(sse_stream, '_last_tick', time.time()))
                sleep_time = max(0, STATS_INTERVAL - elapsed)
                sse_stream._last_tick = time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _collect_stats():
    """Collect all stats in one go."""
    gpus = await get_gpu_stats()
    cpu = await get_cpu_stats()
    mem = await get_memory_stats()
    disk = await get_disk_stats()

    # LLM backend status
    llm_running = await check_llm_running()
    loaded_models = await get_loaded_models() if llm_running else []
    available_models = await get_available_models() if llm_running else []

    # Log parser metrics (works for both LM Studio and Ollama)
    lm_log = await log_parser.get_latest()

    # Fallback: use loaded model name if log parser didn't find one
    if not lm_log.get("model") and loaded_models:
        lm_log["model"] = loaded_models[0]

    # Calculate total power
    total_power = sum(g.get("power_draw", 0) for g in gpus if isinstance(g, dict))

    # Token counts and costs from log parser
    total_input = lm_log.get("total_input_tokens", 0)
    total_output = lm_log.get("total_output_tokens", 0)
    session_input = lm_log.get("session_input_tokens", 0)
    session_output = lm_log.get("session_output_tokens", 0)
    cost_total = (total_input * COST_INPUT_PER_M + total_output * COST_OUTPUT_PER_M) / 1_000_000
    cost_session = (session_input * COST_INPUT_PER_M + session_output * COST_OUTPUT_PER_M) / 1_000_000

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "Ollama" if DASHBOARD_MODE == "ollama" else "LM Studio",
        "gpus": gpus,
        "cpu": cpu,
        "memory": mem,
        "disk": disk,
        "total_power": round(total_power, 1),
        "lm_studio": {
            "running": llm_running,
            "loaded_models": loaded_models,
            "available_models": available_models,
            "active_sessions": [],  # No more session tracking
            "log_stats": lm_log,
            "queue_length": lm_log.get("queue_length", 0),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "session_input_tokens": session_input,
            "session_output_tokens": session_output,
            "cost_total": round(cost_total, 4),
            "cost_session": round(cost_session, 4),
            "cost_input_per_m": COST_INPUT_PER_M,
            "cost_output_per_m": COST_OUTPUT_PER_M,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="info",
    )
