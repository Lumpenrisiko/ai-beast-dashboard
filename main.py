#!/usr/bin/env python3
"""AI Beast Dashboard — FastAPI Backend (Log-only, no proxy)"""

import asyncio
import glob
import json
import os
import re
import subprocess
import time
from collections import deque
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
# Runtime-configurable settings (persisted to config.json)
_CONFIG_FILE = os.path.expanduser("~/.ai-beast-dashboard/config.json")

def _load_config():
    defaults = {
        "mode": os.getenv("DASHBOARD_MODE", "lmstudio").lower(),
        "lm_studio_url": os.getenv("LM_STUDIO_URL", "http://localhost:1234"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "unsloth_metrics_port": int(os.getenv("UNSLOTH_METRICS_PORT", "0")),  # 0 = auto-detect
        "llamacpp_port": int(os.getenv("LLAMACPP_PORT", "8888")),  # 0 = auto-detect
        "cost_input_per_m": float(os.getenv("COST_INPUT_PER_M", "0.325")),
        "cost_output_per_m": float(os.getenv("COST_OUTPUT_PER_M", "1.95")),
        "cost_cache_ratio": float(os.getenv("COST_CACHE_RATIO", "0.10")),
    }
    try:
        with open(_CONFIG_FILE, "r") as f:
            saved = json.load(f)
        for k in defaults:
            if k in saved:
                defaults[k] = saved[k]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults

_DASHBOARD_CONFIG = _load_config()

def _save_config():
    try:
        os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
        with open(_CONFIG_FILE, "w") as f:
            json.dump(_DASHBOARD_CONFIG, f, indent=2)
    except Exception:
        pass

def get_dashboard_mode():
    return _DASHBOARD_CONFIG["mode"]

def set_dashboard_mode(mode: str):
    valid_modes = ("lmstudio", "ollama", "unsloth", "llamacpp")
    mode = mode.lower().strip()
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
    old_mode = _DASHBOARD_CONFIG["mode"]
    _DASHBOARD_CONFIG["mode"] = mode
    _save_config()
    return {"previous": old_mode, "current": mode}

def get_lm_studio_url():
    return _DASHBOARD_CONFIG.get("lm_studio_url", "http://localhost:1234")

def set_lm_studio_url(url: str):
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    old = _DASHBOARD_CONFIG["lm_studio_url"]
    _DASHBOARD_CONFIG["lm_studio_url"] = url
    _save_config()
    return {"previous": old, "current": url}

def get_ollama_url():
    return _DASHBOARD_CONFIG.get("ollama_url", "http://localhost:11434")

def set_ollama_url(url: str):
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    old = _DASHBOARD_CONFIG["ollama_url"]
    _DASHBOARD_CONFIG["ollama_url"] = url
    _save_config()
    return {"previous": old, "current": url}

def get_unsloth_port():
    return int(_DASHBOARD_CONFIG.get("unsloth_metrics_port", 0))

def get_llamacpp_port():
    return int(_DASHBOARD_CONFIG.get("llamacpp_port", 8888))

def set_llamacpp_port(port: int):
    port = max(0, min(65535, int(port)))
    old = _DASHBOARD_CONFIG.get("llamacpp_port", 8888)
    _DASHBOARD_CONFIG["llamacpp_port"] = port
    _save_config()
    return {"previous": old, "current": port}

def set_unsloth_port(port: int):
    port = max(0, min(65535, int(port)))
    old = _DASHBOARD_CONFIG["unsloth_metrics_port"]
    _DASHBOARD_CONFIG["unsloth_metrics_port"] = port
    _save_config()
    return {"previous": old, "current": port}

def get_cost_input_per_m():
    return float(_DASHBOARD_CONFIG.get("cost_input_per_m", 0.325))

def set_cost_input_per_m(cost: float):
    cost = max(0, float(cost))
    old = _DASHBOARD_CONFIG["cost_input_per_m"]
    _DASHBOARD_CONFIG["cost_input_per_m"] = cost
    _save_config()
    return {"previous": old, "current": cost}

def get_cost_output_per_m():
    return float(_DASHBOARD_CONFIG.get("cost_output_per_m", 1.95))

def set_cost_output_per_m(cost: float):
    cost = max(0, float(cost))
    old = _DASHBOARD_CONFIG["cost_output_per_m"]
    _DASHBOARD_CONFIG["cost_output_per_m"] = cost
    _save_config()
    return {"previous": old, "current": cost}

def get_cost_cache_ratio():
    return float(_DASHBOARD_CONFIG.get("cost_cache_ratio", 0.10))

def set_cost_cache_ratio(ratio: float):
    ratio = max(0.0, min(1.0, float(ratio)))
    old = _DASHBOARD_CONFIG["cost_cache_ratio"]
    _DASHBOARD_CONFIG["cost_cache_ratio"] = ratio
    _save_config()
    return {"previous": old, "current": ratio}

def get_all_settings():
    return {
        "mode": _DASHBOARD_CONFIG["mode"],
        "lm_studio_url": _DASHBOARD_CONFIG.get("lm_studio_url", "http://localhost:1234"),
        "ollama_url": _DASHBOARD_CONFIG.get("ollama_url", "http://localhost:11434"),
        "unsloth_metrics_port": int(_DASHBOARD_CONFIG.get("unsloth_metrics_port", 0)),
        "llamacpp_port": int(_DASHBOARD_CONFIG.get("llamacpp_port", 8888)),
        "cost_input_per_m": float(_DASHBOARD_CONFIG.get("cost_input_per_m", 0.325)),
        "cost_output_per_m": float(_DASHBOARD_CONFIG.get("cost_output_per_m", 1.95)),
        "cost_cache_ratio": float(_DASHBOARD_CONFIG.get("cost_cache_ratio", 0.10)),
    }

# Backward compat aliases — always read current value from config dict
class DashboardModeProxy:
    def __eq__(self, other):
        return get_dashboard_mode() == other
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return get_dashboard_mode()
DASHBOARD_MODE = DashboardModeProxy()

# Backward compat URL aliases (used throughout code)
class UrlProxy:
    def __init__(self, getter):
        self._getter = getter
    def __str__(self):
        return self._getter()
    def __repr__(self):
        return str(self)
LM_STUDIO_URL = UrlProxy(get_lm_studio_url)
OLLAMA_URL = UrlProxy(get_ollama_url)

class PortProxy:
    def __init__(self, getter):
        self._getter = getter
    def __int__(self):
        return self._getter()
    def __float__(self):
        return float(self._getter())
    def __gt__(self, other):
        return self._getter() > other
    def __eq__(self, other):
        return self._getter() == other
UNSLOTH_METRICS_PORT = PortProxy(get_unsloth_port)
LLAMACPP_PORT = PortProxy(get_llamacpp_port)
# Logdatei eines direkt gestarteten llama-server. Nur dort steht der echte
# Prompt-Fortschritt; /slots liefert ihn nicht (n_prompt_tokens waechst
# chargenweise mit n_prompt_tokens_processed, das Verhaeltnis haengt dauerhaft
# bei ~99 %). Die Startskripte auf der Arbeitsflaeche schreiben hierhin.
LLAMACPP_LOG = os.getenv(
    "LLAMACPP_LOG", os.path.expanduser("~/llama-logs/llama-server.log")
)

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
STATS_INTERVAL = int(os.getenv("STATS_INTERVAL", "1"))
CHART_HISTORY = 60  # seconds of chart data to keep
# Token pricing (EUR per 1M tokens)
# Backward compat cost proxies (used throughout code)
class CostProxy:
    def __init__(self, getter):
        self._getter = getter
    def __float__(self):
        return float(self._getter())
    def __mul__(self, other):
        return float(self._getter()) * other
    def __rmul__(self, other):
        return other * float(self._getter())
COST_INPUT_PER_M = CostProxy(get_cost_input_per_m)
COST_OUTPUT_PER_M = CostProxy(get_cost_output_per_m)
# Price for KV-cached input tokens as a fraction of the input price
# (OpenRouter convention: cached prompts cost 10% of the input price).
COST_CACHE_RATIO = CostProxy(get_cost_cache_ratio)


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
        # Track previous prompt_progress to detect new request start
        self._prev_prompt_progress = 0
        # Unsloth mode: previous /metrics counter snapshot (None = not seeded yet)
        self._unsloth_prev = None
        self._unsloth_was_active = False
        self._unsloth_model_check = 0
        self._unsloth_req_prompt_base = None
        # llamacpp mode: state for the directly-started llama-server
        self._llamacpp_prev = None
        self._llamacpp_was_active = False
        self._llamacpp_task_id = None
        # Tailing der llama-server-Logdatei fuer den echten Prompt-Fortschritt
        self._llamacpp_log_pos = 0
        self._llamacpp_log_progress = None
        self._llamacpp_log_progress_ts = 0.0
        self._llamacpp_log_pp_rate = None
        self._llamacpp_log_pp_rate_ts = 0.0
        self._llamacpp_log_pp_tokens = None
        self._llamacpp_log_tg_rate = None
        self._llamacpp_log_tg_rate_ts = 0.0
        self._llamacpp_log_draft = None      # (akzeptanz, accepted, generated, mean_len)
        self._llamacpp_log_draft_ts = 0.0
        self._llamacpp_model_check = 0
        self._llamacpp_req_prompt_base = None
        # Freetoken (anderer Server auf demselben Port, ohne /metrics und /slots).
        # Erkannt am 404 von /metrics; dann liefert /v1/stats die Werte.
        self._backend_kind = "llamacpp"      # "llamacpp" | "freetoken"
        self._backend_recheck = 0.0
        self._ft_prev = None                  # letzter /v1/stats-Snapshot
        self._ft_page_size = 0                # echte KV-Page-Size aus /v1/cache/status
        # Live-rate tracking from /slots (n_decoded / n_prompt_tokens_processed
        # grow every poll WHILE a request runs, unlike /metrics which only bumps
        # at request end). Keys: t, decoded, processed, n_prompt
        self._llamacpp_live = {"t": 0.0, "decoded": 0, "processed": 0, "n_prompt": 0}
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
            # total_input_tokens = ALL input tokens incl. KV-cached (OpenRouter-style)
            "total_input_tokens": 0,
            "total_input_cached_tokens": 0,
            "total_output_tokens": 0,
            # Session token counters (resettable via /api/reset-session-tokens)
            "session_input_tokens": 0,
            "session_input_cached_tokens": 0,
            "session_output_tokens": 0,
            # Queue tracking
            "queue_length": 0,
            "last_queue_update": 0,
            # Unsloth real-time status (from requests_processing gauge)
            "is_active": False,
            "phase": "idle",
        }
        self._lock = asyncio.Lock()
        self._watch_task = None
        # Skip token counting during the initial log backfill (LM Studio reads
        # the last 100 KB / Ollama the last 30 s on startup). Without this,
        # persisted counters would be double-counted after a restart.
        self._skip_counting = False
        # Persistent token counters — survive dashboard restarts until manually reset
        self._token_counts_file = os.path.expanduser("~/.ai-beast-dashboard/token_counts.json")
        self._load_persisted_token_counts()
        # TTL in seconds - values expire after this.
        # Unsloth/llama.cpp only publishes its counters when a request phase
        # finishes, so a single scrape has to stay visible much longer than a
        # log line (which arrives continuously while the request runs).
        self._ttl = 15 if get_dashboard_mode() in ("unsloth", "llamacpp") else 5

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

    # --- Persistent token counters (survive restarts until manual reset) ---

    _TOKEN_COUNTER_KEYS = (
        "total_input_tokens",
        "total_input_cached_tokens",
        "total_output_tokens",
        "session_input_tokens",
        "session_input_cached_tokens",
        "session_output_tokens",
    )

    def _load_persisted_token_counts(self):
        """Load token counters from disk (called in __init__)."""
        try:
            with open(self._token_counts_file, "r") as f:
                saved = json.load(f)
            for k in self._TOKEN_COUNTER_KEYS:
                if k in saved:
                    self._latest[k] = max(0, int(saved[k]))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _save_persisted_token_counts(self):
        """Write token counters to disk. Caller must hold self._lock."""
        try:
            os.makedirs(os.path.dirname(self._token_counts_file), exist_ok=True)
            data = {k: int(self._latest[k]) for k in self._TOKEN_COUNTER_KEYS}
            tmp = self._token_counts_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._token_counts_file)
        except Exception:
            pass

    def _reset_token_counters(self, scope: str):
        """Reset counters in place. Caller must hold self._lock.

        scope: 'session' resets only session counters,
               'total' resets everything (total + session).
        """
        if scope == "session":
            for k in ("session_input_tokens", "session_input_cached_tokens", "session_output_tokens"):
                self._latest[k] = 0
        else:
            for k in self._TOKEN_COUNTER_KEYS:
                self._latest[k] = 0
        self._save_persisted_token_counts()

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
        """Continuously read new log lines (LM Studio mode) or poll journalctl (Ollama mode) or scrape metrics (Unsloth mode)."""
        while True:
            try:
                await asyncio.sleep(0.3)
                if DASHBOARD_MODE == "ollama":
                    await self._poll_ollama_logs()
                elif DASHBOARD_MODE == "unsloth":
                    await self._poll_unsloth_metrics()
                elif DASHBOARD_MODE == "llamacpp":
                    await self._poll_llamacpp()
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
            # Only the very first read after process start is a backfill of
            # already-counted history. Later file changes (e.g. new day's log
            # file) contain fresh, not-yet-counted lines and must be counted.
            initial_backfill = self._current_file is None
            self._current_file = latest
            self._last_mtime = mtime
            if initial_backfill:
                # Don't count the last-100 KB backfill — those lines were
                # already counted before the previous restart.
                self._skip_counting = True
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
            finally:
                if initial_backfill:
                    self._skip_counting = False
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
                cmd.extend(["--since", "30 seconds ago"])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            output = stdout.decode()
            if not output:
                return
            # First poll after process start covers the last 30 s of history —
            # those lines were already counted before the previous restart.
            initial_backfill = not self._ollama_cursor
            if initial_backfill:
                self._skip_counting = True
            try:
                for line in output.split("\n"):
                    if line:
                        self._parse_line(line)
            finally:
                if initial_backfill:
                    self._skip_counting = False
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

    @staticmethod
    def _detect_llama_port_from_ps() -> int | None:
        """Find llama-server --port from the process list."""
        import subprocess as sp
        try:
            result = sp.run(["ps", "aux"], capture_output=True, text=True, timeout=3)
            for line in result.stdout.split("\n"):
                if "llama-server" in line:
                    m = re.search(r"--port\s+(\d+)", line)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        return None

    def _get_llamacpp_port(self) -> int | None:
        """Port of the directly-started llama-server (config wins, else auto-detect)."""
        if LLAMACPP_PORT > 0:
            return int(LLAMACPP_PORT)
        return self._detect_llama_port_from_ps()

    async def _get_unsloth_metrics_url(self) -> str:
        """Auto-detect llama-server port from running process or use configured port."""
        if UNSLOTH_METRICS_PORT > 0:
            return f"http://127.0.0.1:{UNSLOTH_METRICS_PORT}/metrics"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ps", "aux",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
            for line in stdout.decode().split("\n"):
                if "llama-server" in line:
                    m = re.search(r"--port\s+(\d+)", line)
                    if m:
                        return f"http://127.0.0.1:{m.group(1)}/metrics"
        except Exception:
            pass
        return ""

    @staticmethod
    def _token_rate(d_tokens: float, d_seconds: float, gauge: float, dt_wall: float) -> float:
        """Tokens/s for the work that completed since the previous scrape.

        llama.cpp bumps its counters only when a request finishes a phase, so
        the wall-clock poll interval is NOT the time the work took - dividing by
        it turns a 40 s prompt eval into a ~100k tok/s reading. The server's own
        time counters (ms precision) are the correct denominator.

        Fallbacks: the per-scrape gauge (llama.cpp resets its metrics bucket on
        every scrape, so it holds the average for exactly this window), then
        wall-clock as a last resort.
        """
        if d_seconds > 0.001:
            return d_tokens / d_seconds
        if gauge > 0:
            return gauge
        return d_tokens / dt_wall

    async def _poll_unsloth_metrics(self):
        """Scrape Prometheus-style metrics from llama.cpp /metrics endpoint (Unsloth Studio)."""
        try:
            url = await self._get_unsloth_metrics_url()
            if not url:
                return
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "--max-time", "3", url,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            text = stdout.decode()
            if not text:
                return

            metrics = {}
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    try:
                        metrics[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

            now = time.time()
            requests_processing = metrics.get("llamacpp:requests_processing", 0)
            requests_deferred = metrics.get("llamacpp:requests_deferred", 0)

            # Cumulative counters: tokens and the time llama.cpp spent on them
            snapshot = {
                "prompt": metrics.get("llamacpp:prompt_tokens_total", 0),
                "prompt_cached": metrics.get("llamacpp:prompt_tokens_cached_total", 0),
                "prompt_s": metrics.get("llamacpp:prompt_seconds_total", 0),
                "gen": metrics.get("llamacpp:tokens_predicted_total", 0),
                "gen_s": metrics.get("llamacpp:tokens_predicted_seconds_total", 0),
                "time": now,
            }
            # Per-scrape average gauges (used as fallback)
            gauge_prompt = metrics.get("llamacpp:prompt_tokens_seconds", 0)
            gauge_gen = metrics.get("llamacpp:predicted_tokens_seconds", 0)

            prev = self._unsloth_prev
            # First scrape, or llama-server restarted (counters went backwards):
            # only seed the baseline. Otherwise the whole server lifetime would
            # be charged to a single 0.3 s poll interval.
            restarted = prev is not None and (
                snapshot["prompt"] < prev["prompt"] or snapshot["gen"] < prev["gen"]
            )
            if prev is None or restarted:
                self._unsloth_prev = snapshot
                self._latest["last_update"] = now
                if restarted:
                    # Server neu gestartet: Counter-Baseline neu setzen, damit
                    # die Deltas nicht negativ werden. Die persistenten Token-
                    # Counter bleiben ERHALTEN (nur manueller Reset setzt sie).
                    self._unsloth_was_active = False
                    self._latest["is_active"] = False
                    self._latest["phase"] = "idle"
                    self._unsloth_req_prompt_base = None
                return

            dt_wall = max(now - prev["time"], 0.001)
            delta_prompt = snapshot["prompt"] - prev["prompt"]
            delta_prompt_cached = max(0, int(snapshot["prompt_cached"] - prev["prompt_cached"]))
            delta_gen = snapshot["gen"] - prev["gen"]
            delta_prompt_s = snapshot["prompt_s"] - prev["prompt_s"]
            delta_gen_s = snapshot["gen_s"] - prev["gen_s"]

            if delta_prompt > 0 or delta_prompt_cached > 0:
                # Rate: only real (non-cached) computation counts for tok/s
                if delta_prompt > 0:
                    rate = self._token_rate(delta_prompt, delta_prompt_s, gauge_prompt, dt_wall)
                    if rate > 0:
                        self._latest["prompt_tokens_per_sec"] = rate
                        self._latest["p_s_time"] = now
                self._latest["prompt_tokens"] = int(delta_prompt)
                self._latest["prompt_time_ms"] = delta_prompt_s * 1000.0
                self._latest["has_timing"] = True
                # The counter only moves once the prompt is fully processed.
                self._latest["prompt_progress"] = 100.0
                # OpenRouter-style: "in" = all input tokens incl. KV-cached
                self._latest["total_input_tokens"] += int(delta_prompt) + delta_prompt_cached
                self._latest["total_input_cached_tokens"] += delta_prompt_cached
                self._latest["session_input_tokens"] += int(delta_prompt) + delta_prompt_cached
                self._latest["session_input_cached_tokens"] += delta_prompt_cached
                self._latest["phase"] = "prompt"
                self._save_persisted_token_counts()

            if delta_gen > 0:
                rate = self._token_rate(delta_gen, delta_gen_s, gauge_gen, dt_wall)
                if rate > 0:
                    self._latest["tokens_per_sec"] = rate
                    self._latest["tok_s_time"] = now
                self._latest["eval_tokens"] = int(delta_gen)
                self._latest["eval_time_ms"] = delta_gen_s * 1000.0
                self._latest["has_timing"] = True
                self._latest["prompt_progress"] = 100.0
                self._latest["total_output_tokens"] += int(delta_gen)
                self._latest["session_output_tokens"] += int(delta_gen)
                self._latest["phase"] = "generation"
                self._save_persisted_token_counts()

            n_tokens_max = metrics.get("llamacpp:n_tokens_max", 0)
            if n_tokens_max > 0:
                self._latest["n_tokens_max"] = int(n_tokens_max)

            is_active = requests_processing > 0 or requests_deferred > 0
            # Neue Anfrage gestartet (rising edge): beginnt mit Prompt-Verarbeitung.
            # Prompt-Counter-Baseline merken, um Phase-Wechsel live zu erkennen.
            if is_active and not self._unsloth_was_active:
                # Baseline = Counterstand VOR der Anfrage (letztes Idle-Snapshot).
                # Ist der Prompt-Counter beim ersten aktiven Poll schon gewachsen,
                # ist der Prompt bereits fertig → direkt "generation".
                self._unsloth_req_prompt_base = prev["prompt"]
                self._latest["phase"] = "prompt"
            self._latest["is_active"] = is_active
            if is_active:
                # Prompt-Counter ist seit Request-Start gewachsen → Prompt fertig,
                # es wird jetzt generiert (gen-Counter erst am Request-Ende).
                if (
                    self._latest.get("phase") == "prompt"
                    and self._unsloth_req_prompt_base is not None
                    and snapshot["prompt"] > self._unsloth_req_prompt_base
                ):
                    self._latest["phase"] = "generation"
                self._latest["queue_length"] = int(requests_processing + requests_deferred)
                self._latest["last_queue_update"] = now
            else:
                self._latest["queue_length"] = 0
                self._unsloth_req_prompt_base = None
                # Idle: Phase zurücksetzen, damit Frontend "IDLE" zeigt.
                # Die letzten Raten bleiben sichtbar (Frontend fade-out).
                self._latest["phase"] = "idle"

            # Idle reset: clear progress/timing flags but keep the last rates visible.
            # The frontend will fade stale values and only show 0 when lm.running=false.
            last_rate = max(self._latest.get("tok_s_time", 0), self._latest.get("p_s_time", 0))
            if not is_active and now - last_rate > self._ttl:
                self._latest["has_timing"] = False
                self._latest["prompt_progress"] = 0

            if now - self._unsloth_model_check > 30:
                await self._fetch_unsloth_model()
                self._unsloth_model_check = now

            self._unsloth_prev = snapshot
            self._unsloth_was_active = is_active
            self._latest["last_update"] = now
        except Exception:
            pass

    # ─── llamacpp mode: directly-started llama-server (/metrics + /slots) ───

    def _reset_llamacpp_log_state(self):
        """Alle aus dem Log gelesenen Werte verwerfen.

        Aufgerufen, wenn ein neuer llama-server-Prozess erkannt wird (die
        /metrics-Zaehler laufen dann rueckwaerts). Auch die Leseposition wird
        zurueckgesetzt, damit eine neu angelegte Logdatei von vorn gelesen wird.
        """
        self._llamacpp_log_pos = 0
        self._llamacpp_log_progress = None
        self._llamacpp_log_progress_ts = 0.0
        self._llamacpp_log_pp_rate = None
        self._llamacpp_log_pp_rate_ts = 0.0
        self._llamacpp_log_pp_tokens = None
        self._llamacpp_log_tg_rate = None
        self._llamacpp_log_tg_rate_ts = 0.0
        self._llamacpp_log_draft = None
        self._llamacpp_log_draft_ts = 0.0
        for k in ("draft_acceptance", "draft_accepted", "draft_generated",
                  "draft_mean_len", "draft_time"):
            self._latest[k] = 0
        self._latest["draft_time"] = 0.0

    def _read_llamacpp_log_progress(self):
        """Echten Prompt-Fortschritt aus der llama-server-Logdatei nachlesen.

        llama.cpp kennt die Gesamtlaenge des Prompts und schreibt sie heraus:
          ... prompt processing, n_tokens = 2694, progress = 0.84, t = 10.16 s / ...

        Ueber /slots ist dieser Wert nicht rekonstruierbar, weil dort
        n_prompt_tokens chargenweise mitwaechst.

        Gelesen wird nur der seit dem letzten Aufruf angewachsene Teil.
        Schrumpft die Datei, wurde sie beim Serverstart neu angelegt -- dann
        wird von vorn gelesen.
        """
        path = LLAMACPP_LOG
        if not path:
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size < self._llamacpp_log_pos:
            self._llamacpp_log_pos = 0
        if size == self._llamacpp_log_pos:
            return
        try:
            with open(path, "r", errors="ignore") as fh:
                fh.seek(self._llamacpp_log_pos)
                chunk = fh.read()
                self._llamacpp_log_pos = fh.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            # Laufende Verarbeitung:
            #   prompt processing, n_tokens = 1474, progress = 0.97,
            #   t = 7.33 s / 201.08 tokens per second
            m = re.search(r"prompt processing.*?progress\s*=\s*([\d.]+)", line)
            if m:
                self._llamacpp_log_progress = min(100.0, float(m.group(1)) * 100.0)
                self._llamacpp_log_progress_ts = time.time()
                m2 = re.search(r"n_tokens\s*=\s*(\d+)", line)
                if m2:
                    self._llamacpp_log_pp_tokens = int(m2.group(1))
                m3 = re.search(r"/\s*([\d.]+)\s*tokens per second", line)
                if m3:
                    self._llamacpp_log_pp_rate = float(m3.group(1))
                    self._llamacpp_log_pp_rate_ts = time.time()
                continue

            # Laufende Generierung:
            #   n_gen = 500, tg = 26.08 t/s, tg_3s = 27.79 t/s
            # Genommen wird tg_3s, das Mittel der letzten drei Sekunden: das
            # Live-Feld soll den Momentanzustand zeigen. Fuer den geglaetteten
            # Wert gibt es die Median-Kachel daneben, die dadurch echte
            # Schwankungen mittelt statt kumulierter Mittelwerte.
            m = re.search(r"n_gen\s*=\s*\d+,.*?tg_3s\s*=\s*([\d.]+)\s*t/s", line)
            if m:
                self._llamacpp_log_tg_rate = float(m.group(1))
                self._llamacpp_log_tg_rate_ts = time.time()
                continue

            # Abschlusszeile der Generierung (ohne "prompt" davor):
            #   eval time = 26514.85 ms / 700 tokens (37.93 ms per token,
            #   26.36 tokens per second)
            if "prompt eval time" not in line:
                m = re.search(
                    r"\beval time\s*=.*?/\s*\d+\s*tokens.*?"
                    r"([\d.]+)\s*tokens per second",
                    line,
                )
                if m:
                    self._llamacpp_log_tg_rate = float(m.group(1))
                    self._llamacpp_log_tg_rate_ts = time.time()
                    continue

            # MTP-Spekulation, je Anfrage am Ende:
            #   draft acceptance = 0.49525 ( 417 accepted / 842 generated),
            #   mean len = 2.48
            # Die /metrics-Zaehler liefern hier nur Summen seit Serverstart,
            # das Log dagegen den Wert der einzelnen Anfrage.
            m = re.search(
                r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*"
                r"(\d+)\s*generated\s*\).*?mean len\s*=\s*([\d.]+)",
                line,
            )
            if m:
                self._llamacpp_log_draft = (
                    float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)),
                )
                self._llamacpp_log_draft_ts = time.time()
                continue

            # Abschlusszeile mit dem maessgeblichen Endwert:
            #   prompt eval time = 10081.60 ms / 1990 tokens
            #   ( 5.07 ms per token, 197.39 tokens per second)
            m = re.search(
                r"prompt eval time\s*=.*?/\s*(\d+)\s*tokens.*?"
                r"([\d.]+)\s*tokens per second",
                line,
            )
            if m:
                self._llamacpp_log_pp_tokens = int(m.group(1))
                self._llamacpp_log_pp_rate = float(m.group(2))
                self._llamacpp_log_pp_rate_ts = time.time()
                self._llamacpp_log_progress = 100.0
                self._llamacpp_log_progress_ts = time.time()

    async def _poll_llamacpp(self):
        """Scrape a directly-started llama-server via /metrics and /slots.

        Provides live token/s, prompt/s, real prompt-processing progress (from
        /slots n_prompt_tokens_processed), queue, phase and MTP speculative-
        decoding statistics. The server must be started with --metrics.
        """
        port = self._get_llamacpp_port()
        if not port:
            self._latest["running"] = False
            return
        # Ist Freetoken bekannt, /metrics und /slots gar nicht erst anfragen --
        # jede Anfrage erzeugt dort eine 404-Zeile im Log (alle 0,3 s).
        # Alle 30 s wird neu geprueft, falls wieder llama-server laeuft.
        if self._backend_kind == "freetoken" and time.time() < self._backend_recheck:
            await self._poll_freetoken(port)
            return
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/metrics",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        text = await resp.text() if resp.status == 200 else ""
                except Exception:
                    text = ""
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/slots",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        slots = await resp.json() if resp.status == 200 else []
                except Exception:
                    slots = []

            if not text:
                # Kein llama-server -- vielleicht Freetoken auf demselben Port
                await self._poll_freetoken(port)
                return
            if self._backend_kind != "llamacpp":
                # Wechsel Freetoken -> llama-server: Zustand neu aufsetzen
                self._backend_kind = "llamacpp"
                self._llamacpp_prev = None
                self._llamacpp_model_check = 0

            metrics = {}
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    try:
                        metrics[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

            now = time.time()
            requests_processing = metrics.get("llamacpp:requests_processing", 0)
            requests_deferred = metrics.get("llamacpp:requests_deferred", 0)

            snapshot = {
                "prompt": metrics.get("llamacpp:prompt_tokens_total", 0),
                "prompt_cached": metrics.get("llamacpp:prompt_tokens_cached_total", 0),
                "prompt_s": metrics.get("llamacpp:prompt_seconds_total", 0),
                "gen": metrics.get("llamacpp:tokens_predicted_total", 0),
                "gen_s": metrics.get("llamacpp:tokens_predicted_seconds_total", 0),
                "draft": metrics.get("llamacpp:spec_decode_num_draft_tokens_total", 0),
                "accepted": metrics.get("llamacpp:spec_decode_num_accepted_tokens_total", 0),
                "time": now,
            }
            gauge_prompt = metrics.get("llamacpp:prompt_tokens_seconds", 0)
            gauge_gen = metrics.get("llamacpp:predicted_tokens_seconds", 0)

            prev = self._llamacpp_prev
            restarted = prev is not None and (
                snapshot["prompt"] < prev["prompt"] or snapshot["gen"] < prev["gen"]
            )
            if prev is None or restarted:
                self._llamacpp_prev = snapshot
                self._latest["last_update"] = now
                if restarted:
                    self._llamacpp_was_active = False
                    self._llamacpp_task_id = None
                    self._llamacpp_req_prompt_base = None
                    self._latest["is_active"] = False
                    self._latest["phase"] = "idle"
                    self._latest["prompt_progress"] = 0
                    # Neuer Serverprozess = neues Modell. Die aus dem Log
                    # gelesenen Werte der letzten Anfrage gehoeren zum ALTEN
                    # Modell und wuerden sonst weiter angezeigt, bis die erste
                    # Anfrage des neuen Modells fertig ist.
                    self._reset_llamacpp_log_state()
                return

            dt_wall = max(now - prev["time"], 0.001)
            d_prompt = snapshot["prompt"] - prev["prompt"]
            d_prompt_cached = max(0, int(snapshot["prompt_cached"] - prev["prompt_cached"]))
            d_gen = snapshot["gen"] - prev["gen"]
            d_prompt_s = snapshot["prompt_s"] - prev["prompt_s"]
            d_gen_s = snapshot["gen_s"] - prev["gen_s"]

            if d_prompt > 0 or d_prompt_cached > 0:
                if d_prompt > 0:
                    rate = self._token_rate(d_prompt, d_prompt_s, gauge_prompt, dt_wall)
                    if rate > 0:
                        self._latest["prompt_tokens_per_sec"] = rate
                        self._latest["p_s_time"] = now
                    self._latest["prompt_tokens"] = int(d_prompt)
                    self._latest["prompt_time_ms"] = d_prompt_s * 1000.0
                self._latest["has_timing"] = True
                # OpenRouter-style: "in" = all input tokens incl. KV-cached
                self._latest["total_input_tokens"] += int(d_prompt) + d_prompt_cached
                self._latest["total_input_cached_tokens"] += d_prompt_cached
                self._latest["session_input_tokens"] += int(d_prompt) + d_prompt_cached
                self._latest["session_input_cached_tokens"] += d_prompt_cached

            if d_gen > 0:
                rate = self._token_rate(d_gen, d_gen_s, gauge_gen, dt_wall)
                if rate > 0:
                    self._latest["tokens_per_sec"] = rate
                    self._latest["tok_s_time"] = now
                self._latest["eval_tokens"] = int(d_gen)
                self._latest["eval_time_ms"] = d_gen_s * 1000.0
                self._latest["has_timing"] = True
                self._latest["total_output_tokens"] += int(d_gen)
                self._latest["session_output_tokens"] += int(d_gen)

            # ── MTP / speculative decoding: Summen seit Serverstart ──
            # Bewusst in eigene *_total-Felder. Die gleichnamigen Felder ohne
            # Suffix tragen den Wert der ZULETZT ABGESCHLOSSENEN Anfrage und
            # werden weiter unten aus dem Log gesetzt.
            draft_total = int(snapshot["draft"])
            accepted_total = int(snapshot["accepted"])
            self._latest["draft_generated_total"] = draft_total
            self._latest["draft_accepted_total"] = accepted_total
            self._latest["draft_acceptance_total"] = (
                (accepted_total / draft_total) if draft_total > 0 else 0.0
            )

            n_tokens_max = metrics.get("llamacpp:n_tokens_max", 0)
            if n_tokens_max > 0:
                self._latest["n_tokens_max"] = int(n_tokens_max)

            is_active = requests_processing > 0 or requests_deferred > 0

            # Echten Fortschritt aus der Logdatei nachziehen (falls vorhanden)
            self._read_llamacpp_log_progress()

            # MTP-Werte der zuletzt abgeschlossenen Anfrage.
            # BEWUSST ausserhalb des is_active-Zweigs: die Draft-Zeile
            # erscheint erst am ENDE einer Anfrage, zu diesem Zeitpunkt ist
            # der Server oft schon wieder idle. Im aktiven Zweig wuerde der
            # Wert dann nie ankommen.
            # Kein Reset bei neuer Anfrage -- sonst staende das Feld waehrend
            # jeder Generierung leer. Geleert wird nur beim Serverwechsel.
            if self._llamacpp_log_draft is not None:
                acc, accepted, generated, mean_len = self._llamacpp_log_draft
                self._latest["draft_acceptance"] = acc
                self._latest["draft_accepted"] = accepted
                self._latest["draft_generated"] = generated
                self._latest["draft_mean_len"] = mean_len
                self._latest["draft_time"] = self._llamacpp_log_draft_ts

            # ── Real prompt-processing progress from /slots ──
            pp_progress = 0.0
            slot_processing = False
            live_decoded = 0
            live_processed = 0
            live_n_prompt = 0
            slot_task_id = None
            if isinstance(slots, list):
                for s in slots:
                    if s.get("is_processing"):
                        slot_processing = True
                        if slot_task_id is None:
                            slot_task_id = s.get("id_task")
                    total = s.get("n_prompt_tokens", 0) or 0
                    if total > 0:
                        processed = (s.get("n_prompt_tokens_processed", 0) or 0) + (
                            s.get("n_prompt_tokens_cache", 0) or 0
                        )
                        pp_progress = max(pp_progress, min(100.0, processed / total * 100.0))
                    live_n_prompt = max(live_n_prompt, total)
                    nt = s.get("next_token")
                    if isinstance(nt, list) and nt and isinstance(nt[0], dict):
                        live_decoded += int(nt[0].get("n_decoded", 0) or 0)
                    live_processed += int(s.get("n_prompt_tokens_processed", 0) or 0)

            # ── Live token rates from /slots deltas ──
            # /metrics counters only bump when a request FINISHES a phase, so
            # during a long generation the dashboard rate looked frozen for
            # many seconds. n_decoded / n_prompt_tokens_processed in /slots
            # grow on every poll WHILE the slot works — divide their delta by
            # the wall time between polls for a near real-time rate. At request
            # end the authoritative /metrics average still overwrites them.
            prev_live = self._llamacpp_live
            if slot_processing and prev_live["t"] > 0 and (now - prev_live["t"]) > 0.15:
                dt_live = now - prev_live["t"]
                d_decoded = live_decoded - prev_live["decoded"]
                if 0 < d_decoded < 10000:
                    rate = d_decoded / dt_live
                    if 0 < rate < 2000:
                        self._latest["tokens_per_sec"] = rate
                        self._latest["tok_s_time"] = now
                        self._latest["has_timing"] = True
                d_processed = live_processed - prev_live["processed"]
                if 0 < d_processed < 100000:
                    rate = d_processed / dt_live
                    if 0 < rate < 20000:
                        self._latest["prompt_tokens_per_sec"] = rate
                        self._latest["p_s_time"] = now
                        self._latest["has_timing"] = True
            elif not slot_processing:
                # Idle: remember baseline so the next request starts fresh
                # (n_decoded restarts at 0 for every request).
                pass
            self._llamacpp_live = {
                "t": now, "decoded": live_decoded,
                "processed": live_processed, "n_prompt": live_n_prompt,
            }

            # Phase detection: neue Anfrage erkennen.
            # Frueher nur ueber die steigende Flanke von is_active. Das
            # scheitert bei Warteschlangen: liegen Anfragen hintereinander an,
            # faellt is_active nie auf False, die Flanke bleibt aus und die
            # Phase haengt auf "generation" der Vorgaengeranfrage fest --
            # der Balken stand dauerhaft auf 100 %.
            # id_task aus /slots wechselt dagegen bei jeder neuen Anfrage.
            new_request = (
                is_active and not self._llamacpp_was_active
            ) or (
                slot_task_id is not None and slot_task_id != self._llamacpp_task_id
            )
            if slot_task_id is not None:
                self._llamacpp_task_id = slot_task_id
            if new_request:
                self._llamacpp_req_prompt_base = prev["prompt"]
                self._latest["phase"] = "prompt"
                self._latest["prompt_progress"] = 0.0
                # Werte der Vorgaengeranfrage verwerfen
                self._llamacpp_log_progress = None
                self._llamacpp_log_progress_ts = 0.0
                self._llamacpp_log_pp_rate = None
                self._llamacpp_log_pp_rate_ts = 0.0
                self._llamacpp_log_pp_tokens = None
                self._llamacpp_log_tg_rate = None
                self._llamacpp_log_tg_rate_ts = 0.0
            self._latest["is_active"] = is_active
            if is_active:
                if self._latest.get("phase") is None:
                    self._latest["phase"] = "prompt"
                # Switch to generation once the prompt counter moved OR the
                # slot progress reached 100 %.
                # n_decoded > 0 heisst: der Slot erzeugt bereits Tokens, der
                # Prompt ist also durch. Das ist die verlaessliche Quelle.
                # Der /metrics-Zaehler springt erst NACH Abschluss einer Phase
                # und ist monoton -- einmal ueber der Baseline, meldete er
                # "fertig" fuer jede weitere Anfrage.
                # pp_progress taugt hier nicht als Kriterium: n_prompt_tokens
                # waechst chargenweise mit, das Verhaeltnis liegt deshalb
                # dauerhaft bei ~99 % (gemessen 98,7 % ueber die ganze
                # Prompt-Verarbeitung eines 160k-Prompts).
                prompt_done = live_decoded > 0
                if self._latest.get("phase") == "prompt" and prompt_done:
                    self._latest["phase"] = "generation"
                # Progress bar reflects PP; once generating it is pinned to 100.
                if self._latest.get("phase") == "generation":
                    self._latest["prompt_progress"] = 100.0
                elif (
                    self._llamacpp_log_progress is not None
                    and now - self._llamacpp_log_progress_ts <= 30
                ):
                    # Echter Wert aus dem Log hat Vorrang. pp_progress aus
                    # /slots ist bei langen Prompts unbrauchbar (~99 % ab der
                    # ersten Charge), taugt aber als Rueckfall, falls keine
                    # Logdatei konfiguriert ist.
                    self._latest["prompt_progress"] = self._llamacpp_log_progress
                elif pp_progress > 0:
                    self._latest["prompt_progress"] = pp_progress

                # Prompt-Rate ebenfalls aus dem Log bevorzugen. Die aus
                # /slots-Deltas errechnete Rate ist unzuverlaessig, weil
                # n_prompt_tokens_processed chargenweise springt: zwischen
                # zwei Polls entweder 0 oder ein ganzer Batch. llama.cpp
                # rechnet die Rate dagegen ueber die echte Laufzeit.
                if (
                    self._llamacpp_log_tg_rate is not None
                    and now - self._llamacpp_log_tg_rate_ts <= 30
                ):
                    self._latest["tokens_per_sec"] = self._llamacpp_log_tg_rate
                    self._latest["tok_s_time"] = self._llamacpp_log_tg_rate_ts
                    self._latest["has_timing"] = True

                if (
                    self._llamacpp_log_pp_rate is not None
                    and now - self._llamacpp_log_pp_rate_ts <= 30
                ):
                    self._latest["prompt_tokens_per_sec"] = self._llamacpp_log_pp_rate
                    self._latest["p_s_time"] = self._llamacpp_log_pp_rate_ts
                    self._latest["has_timing"] = True
                    if self._llamacpp_log_pp_tokens is not None:
                        self._latest["prompt_tokens"] = self._llamacpp_log_pp_tokens

                self._latest["queue_length"] = int(requests_processing + requests_deferred)
                self._latest["last_queue_update"] = now
            else:
                # If the slot is still busy but metrics gauge says idle (rare),
                # trust the slot for the progress bar.
                if slot_processing and pp_progress > 0:
                    self._latest["prompt_progress"] = pp_progress
                else:
                    self._latest["prompt_progress"] = 0
                self._latest["queue_length"] = 0
                self._llamacpp_req_prompt_base = None
                self._latest["phase"] = "idle"

            if now - self._llamacpp_model_check > 30:
                await self._fetch_llamacpp_model(port)
                self._llamacpp_model_check = now

            self._llamacpp_prev = snapshot
            self._llamacpp_was_active = is_active
            self._latest["last_update"] = now
        except Exception:
            pass

    async def _poll_freetoken(self, port: int) -> bool:
        """Freetoken-Server ueber /v1/stats abfragen.

        Freetoken hat weder /metrics noch /slots. /v1/stats liefert:
          throughput.decode_tps / prefill_tps -- rollierendes Fenster, live
          requests.active / completed, prompt_tokens_total, completion_tokens_total
          kv.used_pages -- Kontextfuellung (Page-Size NICHT aus /v1/stats nehmen,
                          dort steht 1; die echte steht in /v1/cache/status)
        Nicht verfuegbar: Gesamtlaenge des laufenden Prompts (-> kein
        Fortschrittsbalken), MTP (Freetoken nutzt keine spekulative Dekodierung).
        Liefert False, wenn unter dem Port kein Freetoken antwortet.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{port}/v1/stats",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status != 200:
                        return False
                    st = await resp.json()
                if not isinstance(st, dict) or "throughput" not in st:
                    return False
                inst = st.get("instance_id")
                prev = self._ft_prev
                new_instance = (
                    self._backend_kind != "freetoken"
                    or prev is None
                    or prev.get("instance_id") != inst
                )
                if new_instance or not self._ft_page_size:
                    try:
                        async with session.get(
                            f"http://127.0.0.1:{port}/v1/cache/status",
                            timeout=aiohttp.ClientTimeout(total=3),
                        ) as resp:
                            if resp.status == 200:
                                geo = (await resp.json()).get("geometry", {}) or {}
                                self._ft_page_size = int(geo.get("page_size", 0) or 0)
                    except Exception:
                        pass
        except Exception:
            return False

        now = time.time()
        self._backend_kind = "freetoken"
        self._backend_recheck = now + 30
        req = st.get("requests", {}) or {}
        thr = st.get("throughput", {}) or {}
        model = st.get("model", {}) or {}
        snap = {
            "instance_id": inst,
            "prompt": int(req.get("prompt_tokens_total", 0) or 0),
            "gen": int(req.get("completion_tokens_total", 0) or 0),
            "completed": int(req.get("completed", 0) or 0),
            "active": int(req.get("active", 0) or 0),
        }

        if new_instance:
            # Neuer Serverprozess = neues Modell: Werte der alten Instanz verwerfen.
            # Zaehler NICHT nachtragen -- die Summen seit Serverstart wurden
            # (falls das Dashboard schon lief) bereits gezaehlt.
            self._ft_prev = snap
            self._reset_llamacpp_log_state()
            for k in ("draft_acceptance", "draft_accepted", "draft_generated",
                      "draft_generated_total", "draft_accepted_total",
                      "draft_acceptance_total"):
                self._latest[k] = 0
            self._latest["prompt_progress"] = 0
            self._latest["phase"] = "idle"
            self._latest["is_active"] = False
            self._latest["model"] = model.get("id", "") or ""
            self._latest["last_update"] = now
            return True

        d_prompt = max(0, snap["prompt"] - prev["prompt"])
        d_gen = max(0, snap["gen"] - prev["gen"])

        is_active = snap["active"] > 0
        new_request = is_active and (
            not prev["active"] or snap["completed"] > prev["completed"]
        )
        if new_request:
            self._latest["phase"] = "prompt"
            self._latest["prompt_progress"] = 0.0
            self._latest["prompt_tokens"] = 0

        if d_prompt > 0:
            # prefill_tps ist ein Mittel ueber ein rollierendes Fenster --
            # dieselbe Groesse, die Freetoken im Log als "input throughput" meldet.
            self._latest["prompt_tokens_per_sec"] = float(thr.get("prefill_tps", 0) or 0)
            self._latest["p_s_time"] = now
            self._latest["prompt_tokens"] = int(self._latest.get("prompt_tokens", 0) or 0) + d_prompt
            self._latest["has_timing"] = True
            self._latest["total_input_tokens"] += d_prompt
            self._latest["session_input_tokens"] += d_prompt
        if d_gen > 0:
            self._latest["tokens_per_sec"] = float(thr.get("decode_tps", 0) or 0)
            self._latest["tok_s_time"] = now
            self._latest["eval_tokens"] = d_gen
            self._latest["has_timing"] = True
            self._latest["total_output_tokens"] += d_gen
            self._latest["session_output_tokens"] += d_gen

        if is_active:
            if d_gen > 0:
                self._latest["phase"] = "generation"
            if self._latest.get("phase") == "generation":
                self._latest["prompt_progress"] = 100.0
            self._latest["queue_length"] = snap["active"]
            self._latest["last_queue_update"] = now
        else:
            self._latest["phase"] = "idle"
            self._latest["prompt_progress"] = 0
            self._latest["queue_length"] = 0
        self._latest["is_active"] = is_active

        kv = st.get("kv", {}) or {}
        if self._ft_page_size and kv.get("used_pages") is not None:
            self._latest["n_tokens_max"] = int(kv["used_pages"]) * self._ft_page_size
        self._latest["model"] = model.get("id", "") or self._latest.get("model", "")

        self._ft_prev = snap
        self._latest["last_update"] = now
        return True

    async def _fetch_llamacpp_model(self, port: int):
        """Model alias/name from the llama-server /props + /v1/models."""
        try:
            async with aiohttp.ClientSession() as session:
                model = ""
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/props",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            props = await resp.json()
                            model = props.get("model_alias", "") or ""
                except Exception:
                    pass
                if not model:
                    async with session.get(
                        f"http://127.0.0.1:{port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            items = data.get("data") or data.get("models") or []
                            if items:
                                model = items[0].get("id") or items[0].get("name", "")
                if model:
                    self._latest["model"] = model
        except Exception:
            pass

    async def _fetch_unsloth_model(self):
        """Fetch model name from llama.cpp /v1/models endpoint."""
        try:
            if UNSLOTH_METRICS_PORT > 0:
                port = UNSLOTH_METRICS_PORT
            else:
                proc = await asyncio.create_subprocess_exec(
                    "ps", "aux",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
                port = None
                for line in stdout.decode().split("\n"):
                    if "llama-server" in line:
                        m = re.search(r"--port\s+(\d+)", line)
                        if m:
                            port = int(m.group(1))
                            break
            if not port:
                return
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/v1/models",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            data = json.loads(stdout.decode())
            models = data.get("data", []) or data.get("models", [])
            if models:
                model_id = models[0].get("id", models[0].get("name", ""))
                self._latest["model"] = model_id
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
            new_progress = float(m.group(1))
            # Reset MTP draft stats when prompt processing starts (transition from idle)
            if new_progress > 0 and self._prev_prompt_progress <= 0:
                self._latest["draft_acceptance"] = 0
                self._latest["draft_accepted"] = 0
                self._latest["draft_generated"] = 0
                self._latest["draft_rate"] = 0
                self._latest["draft_tokens_generated"] = 0
                self._latest["draft_tokens_accepted"] = 0
            self._prev_prompt_progress = new_progress
            self._latest["prompt_progress"] = new_progress
            self._latest["last_update"] = time.time()
            return

        # REAL-TIME: prompt processing, n_tokens = 57344, progress = 0.52, t = 49.08 s / 1168.47 tokens per second
        m = re.search(r"prompt processing.*?n_tokens\s*=\s*(\d+).*?progress\s*=\s*([\d.]+).*?t\s*=\s*[\d.]+\s*s\s*/\s*([\d.]+)\s*tokens per second", line)
        if m:
            new_progress = float(m.group(2)) * 100
            # Reset MTP draft stats when prompt processing starts (transition from idle)
            if new_progress > 0 and self._prev_prompt_progress <= 0:
                self._latest["draft_acceptance"] = 0
                self._latest["draft_accepted"] = 0
                self._latest["draft_generated"] = 0
                self._latest["draft_rate"] = 0
                self._latest["draft_tokens_generated"] = 0
                self._latest["draft_tokens_accepted"] = 0
            self._prev_prompt_progress = new_progress
            self._latest["prompt_tokens"] = int(m.group(1))
            self._latest["prompt_progress"] = new_progress
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
            # Track cumulative input tokens (both total and session).
            # Skipped during the initial log backfill — those lines were
            # already counted before the last restart (persisted counters).
            if not self._skip_counting:
                n_tokens = int(m.group(2))
                self._latest["total_input_tokens"] += n_tokens
                self._latest["session_input_tokens"] += n_tokens
                self._save_persisted_token_counts()
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
            # Track cumulative output tokens (both total and session),
            # skipped during the initial log backfill (see input above).
            if not self._skip_counting:
                n_tokens = int(m.group(2))
                self._latest["total_output_tokens"] += n_tokens
                self._latest["session_output_tokens"] += n_tokens
                self._save_persisted_token_counts()
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

        # Unsloth Studio JSON log: engine_stats with gen_tok_s and prompt_tok_s
        if line.startswith("{") and "engine_stats" in line:
            try:
                entry = json.loads(line)
                if entry.get("event") == "engine_stats":
                    gen_tok_s = entry.get("gen_tok_s", 0)
                    prompt_tok_s = entry.get("prompt_tok_s", 0)
                    running = entry.get("running", 0)
                    
                    was_active = self._latest.get("has_timing", False)
                    is_active = running > 0
                    
                    # Reset draft stats on new request start
                    if is_active and not was_active:
                        self._latest["draft_acceptance"] = 0
                        self._latest["draft_accepted"] = 0
                        self._latest["draft_generated"] = 0
                        self._latest["draft_rate"] = 0
                    
                    if gen_tok_s and gen_tok_s > 0:
                        self._latest["tokens_per_sec"] = gen_tok_s
                        self._latest["tok_s_time"] = time.time()
                        self._latest["has_timing"] = True
                        self._latest["prompt_progress"] = 100.0
                    
                    if prompt_tok_s and prompt_tok_s > 0:
                        self._latest["prompt_tokens_per_sec"] = prompt_tok_s
                        self._latest["p_s_time"] = time.time()
                        self._latest["has_timing"] = True
                        if not (gen_tok_s and gen_tok_s > 0):
                            self._latest["prompt_progress"] = min(self._latest.get("prompt_progress", 50), 99)
                    
                    self._latest["last_update"] = time.time()
            except (json.JSONDecodeError, KeyError):
                pass

        # Unsloth Studio JSON log: model_loaded event
        if line.startswith("{") and "model_loaded" in line:
            try:
                entry = json.loads(line)
                if entry.get("event") == "model_loaded":
                    self._latest["model"] = entry.get("model", "")
            except (json.JSONDecodeError, KeyError):
                pass

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

            # Context-based zeroing (log-based modes only):
            # - When progress == 100% (generation): p/s should be 0
            # - When progress < 100% (prompt processing): tok/s should be 0
            # In Unsloth mode both rates are measured independently from the
            # /metrics counters and progress is always 100 once a prompt has
            # been processed - applying this here would permanently hide p/s.
            if DASHBOARD_MODE != "unsloth":
                if progress >= 100:
                    result["prompt_tokens_per_sec"] = 0
                if progress > 0 and progress < 100:
                    result["tokens_per_sec"] = 0

            # Reset prompt_progress if idle (also reset draft stats + tracking)
            if now - result["last_update"] > 10 and not result["has_timing"]:
                result["prompt_progress"] = 0
                self._prev_prompt_progress = 0
                # Reset MTP draft stats when idle (no active request)
                result["draft_acceptance"] = 0
                result["draft_accepted"] = 0
                result["draft_generated"] = 0
                result["draft_rate"] = 0
                result["draft_tokens_generated"] = 0
                result["draft_tokens_accepted"] = 0

            # Queue TTL: reset if no update for 15s
            if now - result.get("last_queue_update", 0) > 15:
                result["queue_length"] = 0

            # Remove internal tracking fields from API output
            result.pop("last_queue_update", None)
            # Keep tok_s_time/p_s_time for frontend fade effect (stale value indicator)

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


# CPU package power via Intel/AMD RAPL energy counter (µJ).
# energy_uj is world-readable after the udev rule 99-rapl-power.rules.
_RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
_rapl_state: dict = {"energy": None, "time": None}


def _read_cpu_power_w() -> float | None:
    """CPU package power in watts from the RAPL energy counter delta."""
    try:
        with open(_RAPL_ENERGY_PATH) as f:
            energy = int(f.read().strip())
        now = time.monotonic()
        last_e, last_t = _rapl_state["energy"], _rapl_state["time"]
        _rapl_state["energy"] = energy
        _rapl_state["time"] = now
        if last_e is None or last_t is None:
            return None
        de = energy - last_e
        dt = now - last_t
        if de < 0 or dt <= 0:
            # Counter wrapped/reset — skip this sample
            return None
        return round(de / dt / 1_000_000, 1)
    except (OSError, ValueError):
        return None


def _read_cpu_freq_mhz() -> float | None:
    """Average current core frequency (MHz) from cpufreq sysfs.

    psutil.cpu_freq() reports bogus values on some AMD systems, so read the
    per-core scaling_cur_freq directly and average over all cores.
    """
    freqs = []
    try:
        for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
            try:
                with open(p) as f:
                    freqs.append(int(f.read().strip()))
            except (OSError, ValueError):
                continue
    except Exception:
        pass
    if not freqs:
        cpu_freq = psutil.cpu_freq()
        return round(cpu_freq.current, 0) if cpu_freq else None
    return round(sum(freqs) / len(freqs) / 1000.0, 0)


async def get_cpu_stats() -> dict:
    """Get CPU stats."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
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
        "freq_current": _read_cpu_freq_mhz(),
        "power_w": _read_cpu_power_w(),
        "temperature": temp,
        "cores": cpu_count,
    }


# ─── tok/s-Historie fuer den Medianwert ────────────────────────────
# Der Momentanwert schwankt stark: bei identischer Konfiguration wurden 69 bis
# 93 tok/s gemessen, und innerhalb einer Generierung steigt er typisch von ~63
# auf ~87 an (Warmlaufen und schwankende MTP-Akzeptanz). Der Median ueber die
# letzten Minuten ist deshalb deutlich aussagekraeftiger als der letzte
# Messpunkt allein.
_TOK_S_HISTORY: deque = deque(maxlen=600)
_TOK_S_LAST_TS = 0.0
_TOK_S_MODEL = None
_TOK_S_WINDOW_S = 600.0  # Median ueber die letzten 10 Minuten


def _update_tok_s_history(stats: dict) -> None:
    """Neue tok/s-Messung aufnehmen und Median in stats eintragen.

    Aufgenommen wird nur, wenn der Zeitstempel neuer ist als der zuletzt
    gesehene -- der Parser liefert denselben Messwert sonst mehrfach und
    wuerde den Median verzerren.

    Bei einem Modellwechsel wird die Historie verworfen, damit sich Werte
    verschiedener Modelle nicht vermischen (GLM-5.3 Q2 liegt bei ~4 tok/s,
    Qwen Q4 bei ~85 -- ein gemeinsamer Median waere sinnlos).
    """
    global _TOK_S_LAST_TS, _TOK_S_MODEL

    model = stats.get("model")
    if model and model != _TOK_S_MODEL:
        _TOK_S_MODEL = model
        _TOK_S_HISTORY.clear()
        _TOK_S_LAST_TS = 0.0

    ts = float(stats.get("tok_s_time") or 0)
    rate = float(stats.get("tokens_per_sec") or 0)
    if rate > 0 and ts > _TOK_S_LAST_TS:
        _TOK_S_LAST_TS = ts
        _TOK_S_HISTORY.append((ts, rate))

    cutoff = time.time() - _TOK_S_WINDOW_S
    values = sorted(r for t, r in _TOK_S_HISTORY if t >= cutoff)
    if values:
        n = len(values)
        median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        stats["tokens_per_sec_median"] = round(median, 1)
        stats["tokens_per_sec_samples"] = n
    else:
        stats["tokens_per_sec_median"] = None
        stats["tokens_per_sec_samples"] = 0


# Prozesse, deren dateigestuetzter Speicher als "Modell im RAM" zaehlt
_LLM_PROC_PATTERNS = ("llama-server", "llama-cli", "ollama", "lm-studio", "LM Studio")

# Unterhalb dieser Groesse ist RssFile nur Programmcode und Bibliotheken,
# kein Modell. 512 MiB liegt deutlich ueber dem, was die Binaries belegen.
_LLM_MIN_FILE_RSS = 512 * 1024 * 1024


def _llm_model_ram() -> tuple[int, list]:
    """Dateigestuetzter Speicher der LLM-Serverprozesse in Bytes.

    llama.cpp laedt die Gewichte nicht in eigenen Speicher, sondern bildet die
    GGUF-Shards per mmap ab. Solche Seiten sind dateigestuetzt und zaehlen im
    Kernel als Seitencache, nicht als "used" -- psutil.virtual_memory().used
    blendet sie deshalb komplett aus. Ein 158-GiB-Modell erscheint dort gar
    nicht, obwohl es den Speicher belegt.

    Massgeblich ist stattdessen RssFile aus /proc/<pid>/status: der Anteil des
    Prozesses, der dateigestuetzt und aktuell resident ist.
    """
    total = 0
    procs = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmd = " ".join(proc.info.get("cmdline") or ())
            if not any(p in name or p in cmd for p in _LLM_PROC_PATTERNS):
                continue
            with open(f"/proc/{proc.info['pid']}/status") as fh:
                for line in fh:
                    if line.startswith("RssFile:"):
                        nbytes = int(line.split()[1]) * 1024
                        if nbytes >= _LLM_MIN_FILE_RSS:
                            total += nbytes
                            procs.append({
                                "pid": proc.info["pid"],
                                "name": name,
                                "gb": round(nbytes / (1024**3), 1),
                            })
                        break
        except (OSError, ValueError, psutil.Error):
            continue
    return total, procs


async def get_memory_stats() -> dict:
    """Get RAM stats.

    Neben dem klassischen "used" von psutil wird der per mmap belegte
    Modellspeicher separat ausgewiesen, siehe _llm_model_ram().
    """
    mem = psutil.virtual_memory()
    model_bytes, model_procs = _llm_model_ram()

    # Modellseiten liegen im Seitencache und sind in mem.used NICHT enthalten,
    # duerfen also addiert werden, ohne doppelt zu zaehlen.
    used_total = min(mem.used + model_bytes, mem.total)

    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "used_gb": round(mem.used / (1024**3), 1),
        "percent": mem.percent,
        # Neu: Modell im RAM
        "model_gb": round(model_bytes / (1024**3), 1),
        "model_procs": model_procs,
        "used_total_gb": round(used_total / (1024**3), 1),
        "percent_total": round(100 * used_total / mem.total, 1) if mem.total else 0.0,
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
        if DASHBOARD_MODE == "ollama":
            url = f"{OLLAMA_URL}/v1/models"
        elif DASHBOARD_MODE == "unsloth":
            # Use configured port or auto-detect from process
            if UNSLOTH_METRICS_PORT > 0:
                url = f"http://127.0.0.1:{UNSLOTH_METRICS_PORT}/health"
            else:
                import subprocess as sp
                result = sp.run(["ps", "aux"], capture_output=True, text=True, timeout=3)
                port = None
                for line in result.stdout.split("\n"):
                    if "llama-server" in line:
                        m = re.search(r"--port\s+(\d+)", line)
                        if m:
                            port = int(m.group(1))
                            break
                if not port:
                    return False
                url = f"http://127.0.0.1:{port}/health"
        elif DASHBOARD_MODE == "llamacpp":
            port = log_parser._get_llamacpp_port()
            if not port:
                return False
            url = f"http://127.0.0.1:{port}/health"
        else:
            url = f"{LM_STUDIO_URL}/v1/models"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def get_loaded_models() -> list:
    """Get list of models currently loaded in VRAM (actively running)."""
    models = []
    try:
        if DASHBOARD_MODE == "ollama":
            url = str(OLLAMA_URL)
        elif DASHBOARD_MODE in ("unsloth", "llamacpp"):
            port = (
                log_parser._get_llamacpp_port()
                if DASHBOARD_MODE == "llamacpp"
                else (int(UNSLOTH_METRICS_PORT) if UNSLOTH_METRICS_PORT > 0 else log_parser._detect_llama_port_from_ps())
            )
            if not port:
                return []
            url = f"http://127.0.0.1:{port}"
        else:
            url = str(LM_STUDIO_URL)
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
                    elif DASHBOARD_MODE in ("unsloth", "llamacpp"):
                        # llama-server /v1/models: geladene Modelle (OpenAI-Format)
                        for m in (data.get("data") or data.get("models") or []):
                            mid = m.get("id") or m.get("name", "")
                            if mid:
                                models.append(mid)
                    # LM Studio: kein "loaded"-Endpoint (via Log-Parser)
    except Exception:
        pass
    return models


async def get_available_models() -> list:
    """Get list of all available models on disk."""
    models = []
    try:
        if DASHBOARD_MODE in ("unsloth", "llamacpp"):
            port = (
                log_parser._get_llamacpp_port()
                if DASHBOARD_MODE == "llamacpp"
                else (int(UNSLOTH_METRICS_PORT) if UNSLOTH_METRICS_PORT > 0 else log_parser._detect_llama_port_from_ps())
            )
            if not port:
                return []
            url = f"http://127.0.0.1:{port}"
        else:
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
        log_parser._reset_token_counters("session")
    return {"success": True}


@app.post("/api/reset-total-tokens")
async def api_reset_total_tokens():
    """Reset both total and session token counters to zero (and persist)."""
    async with log_parser._lock:
        log_parser._reset_token_counters("total")
    return {"success": True}


@app.post("/api/set-mode")
async def api_set_mode(mode: str):
    """Switch LLM data source at runtime (lmstudio, ollama, unsloth)."""
    try:
        result = set_dashboard_mode(mode)
        # Reset parser state so new mode starts clean
        async with log_parser._lock:
            log_parser._latest["tokens_per_sec"] = 0
            log_parser._latest["prompt_tokens_per_sec"] = 0
            log_parser._latest["has_timing"] = False
            log_parser._latest["prompt_progress"] = 0
            log_parser._latest["tok_s_time"] = 0
            log_parser._latest["p_s_time"] = 0
            # Reset mode-specific state
            if result["current"] != "unsloth":
                log_parser._unsloth_prev = None
                log_parser._unsloth_was_active = False
                log_parser._unsloth_req_prompt_base = None
            if result["current"] != "llamacpp":
                log_parser._llamacpp_prev = None
                log_parser._llamacpp_was_active = False
                log_parser._llamacpp_req_prompt_base = None
                log_parser._llamacpp_model_check = 0
            if result["current"] != "ollama":
                log_parser._ollama_cursor = ""
        return {"success": True, **result}
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/settings")
async def api_get_settings():
    """Get current runtime settings (mode, URLs, ports)."""
    return get_all_settings()


@app.post("/api/set-lm-studio-url")
async def api_set_lm_studio_url(url: str):
    """Set LM Studio server URL at runtime."""
    try:
        result = set_lm_studio_url(url)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-ollama-url")
async def api_set_ollama_url(url: str):
    """Set Ollama server URL at runtime."""
    try:
        result = set_ollama_url(url)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-unsloth-port")
async def api_set_unsloth_port(port: int):
    """Set Unsloth Studio llama-server port at runtime (0 = auto-detect)."""
    try:
        result = set_unsloth_port(port)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-llamacpp-port")
async def api_set_llamacpp_port(port: int):
    """Set llama.cpp server port at runtime (0 = auto-detect from process list)."""
    try:
        result = set_llamacpp_port(port)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-cost-input")
async def api_set_cost_input(cost: float):
    """Set input token price (EUR per 1M tokens)."""
    try:
        result = set_cost_input_per_m(cost)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-cost-output")
async def api_set_cost_output(cost: float):
    """Set output token price (EUR per 1M tokens)."""
    try:
        result = set_cost_output_per_m(cost)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-cost-cache-ratio")
async def api_set_cost_cache_ratio(ratio: float):
    """Set KV-cached input price as fraction of input price (OpenRouter: 0.10)."""
    try:
        result = set_cost_cache_ratio(ratio)
        return {"success": True, **result}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


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

    # tok/s-Median fortschreiben (nach dem Modell-Fallback, damit der
    # Modellwechsel zuverlaessig erkannt wird)
    _update_tok_s_history(lm_log)

    # Calculate total power
    total_power = sum(g.get("power_draw", 0) for g in gpus if isinstance(g, dict))

    # Token counts and costs from log parser
    total_input = lm_log.get("total_input_tokens", 0)
    total_input_cached = lm_log.get("total_input_cached_tokens", 0)
    total_output = lm_log.get("total_output_tokens", 0)
    session_input = lm_log.get("session_input_tokens", 0)
    session_input_cached = lm_log.get("session_input_cached_tokens", 0)
    session_output = lm_log.get("session_output_tokens", 0)
    # OpenRouter-style pricing: all input tokens count, but KV-cached input
    # tokens are billed at cost_cache_ratio (default 10%) of the input price.
    cache_price = float(COST_INPUT_PER_M) * float(COST_CACHE_RATIO)
    cost_total = (
        (total_input - total_input_cached) * COST_INPUT_PER_M
        + total_input_cached * cache_price
        + total_output * COST_OUTPUT_PER_M
    ) / 1_000_000
    cost_session = (
        (session_input - session_input_cached) * COST_INPUT_PER_M
        + session_input_cached * cache_price
        + session_output * COST_OUTPUT_PER_M
    ) / 1_000_000

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": ("Freetoken" if str(DASHBOARD_MODE) == "llamacpp" and log_parser._backend_kind == "freetoken"
                 else {"unsloth": "Unsloth Studio", "ollama": "Ollama", "llamacpp": "llama.cpp"}.get(str(DASHBOARD_MODE), "LM Studio")),
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
            "total_input_cached_tokens": total_input_cached,
            "total_output_tokens": total_output,
            "session_input_tokens": session_input,
            "session_input_cached_tokens": session_input_cached,
            "session_output_tokens": session_output,
            "cost_total": round(cost_total, 4),
            "cost_session": round(cost_session, 4),
            "cost_input_per_m": float(COST_INPUT_PER_M),
            "cost_output_per_m": float(COST_OUTPUT_PER_M),
            "cost_cache_ratio": float(COST_CACHE_RATIO),
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="info",
    )
