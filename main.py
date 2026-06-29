#!/usr/bin/env python3
"""AI Beast Dashboard — FastAPI Backend"""

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import aiohttp
import psutil
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ─── Configuration (from environment variables) ───
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8083"))
DASHBOARD_MODE = os.getenv("DASHBOARD_MODE", "lmstudio").lower()  # "lmstudio" or "ollama"
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
LM_STUDIO_LOG_DIR = os.getenv("LM_STUDIO_LOG_DIR", "")
LACT_ENABLED = os.getenv("LACT_ENABLED", "true").lower() == "true"
STATS_INTERVAL = int(os.getenv("STATS_INTERVAL", "2"))
CHART_HISTORY = 60  # seconds of chart data to keep
# Token pricing (EUR per 1M tokens)
COST_INPUT_PER_M = float(os.getenv("COST_INPUT_PER_M", "0.325"))
COST_OUTPUT_PER_M = float(os.getenv("COST_OUTPUT_PER_M", "1.95"))


class LmStudioLogParser:
    """Parse LM Studio server logs in real-time for accurate metrics."""

    def __init__(self):
        self._current_file = None
        self._position = 0
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
        }
        self._lock = asyncio.Lock()
        self._watch_task = None
        # TTL in seconds - values expire after this
        self._ttl = 5

    async def start_watching(self):
        """Start watching the log file for new entries."""
        self._watch_task = asyncio.ensure_future(self._watch_loop())

    async def stop_watching(self):
        """Stop watching the log file."""
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
                # Extract number from filename like "2026-06-23.20.log"
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
        """Continuously read new log lines."""
        while True:
            try:
                await asyncio.sleep(0.3)  # Check every 300ms
                latest = self._get_latest_log()
                if not latest:
                    continue

                # Always re-read last 200 lines on file change or mtime change
                try:
                    mtime = os.path.getmtime(latest)
                    fsize = os.path.getsize(latest)
                except OSError:
                    mtime = 0
                    fsize = 0

                file_changed = (latest != self._current_file or mtime != getattr(self, '_last_mtime', 0))
                if file_changed:
                    self._current_file = latest
                    self._last_mtime = mtime
                    try:
                        with open(latest, "r") as f:
                            # Seek to last 200 lines (estimate: 200 * 500 bytes = 100KB)
                            seek_pos = max(0, fsize - 100000)
                            f.seek(seek_pos)
                            content = f.read()
                            lines = content.split("\n")
                            # Only parse from the first complete line
                            start_idx = 0
                            if seek_pos > 0 and lines and not lines[0].startswith("["):
                                start_idx = 1
                            # Parse with skip_ttl=True to avoid updating TTL for old lines
                            # BUT we need to parse timing values from recent lines
                            for line in lines[start_idx:]:
                                if line:
                                    # Only skip TTL for progress/model, not for timing
                                    self._parse_line(line, skip_ttl=False)
                            self._position = fsize
                    except (FileNotFoundError, PermissionError):
                        pass
                    continue

                # Read new lines appended since last check
                try:
                    with open(latest, "r") as f:
                        f.seek(self._position)
                        new = f.read()
                        self._position = f.tell()
                        for line in new.split("\n"):
                            if line:
                                self._parse_line(line, skip_ttl=False)
                except (FileNotFoundError, PermissionError):
                    self._current_file = None
                    self._position = 0

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _parse_line(self, line: str, skip_ttl: bool = False):
        """Parse a single log line for metrics."""
        # Prompt processing progress: 55.8%
        m = re.search(r"Prompt processing progress:\s*([\d.]+)%", line)
        if m:
            self._latest["prompt_progress"] = float(m.group(1))
            self._latest["last_update"] = time.time()
            return

        # Model name from progress line
        m = re.search(r"\[INFO\]\[(\S+)\]\s+Prompt processing", line)
        if m:
            self._latest["model"] = m.group(1)

        # REAL-TIME: prompt processing, n_tokens = 57344, progress = 0.52, t = 49.08 s / 1168.47 tokens per second
        m = re.search(r"prompt processing.*?n_tokens\s*=\s*(\d+).*?progress\s*=\s*([\d.]+).*?t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+)\s*tokens per second", line)
        if m:
            self._latest["prompt_tokens"] = int(m.group(1))
            self._latest["prompt_progress"] = float(m.group(2)) * 100
            self._latest["prompt_tokens_per_sec"] = float(m.group(4))
            self._latest["p_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            return

        # REAL-TIME: n_decoded = 100, tg = 54.36 t/s (token generation speed)
        m = re.search(r"n_decoded\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s", line)
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
            if not skip_ttl:
                self._latest["p_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            return

        # eval time = 2209.21 ms / 181 tokens (12.21 ms per token, 81.93 tokens per second)
        m = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(.*?,\s*([\d.]+)\s*tokens per second\)", line)
        if m:
            self._latest["eval_time_ms"] = float(m.group(1))
            self._latest["eval_tokens"] = int(m.group(2))
            self._latest["tokens_per_sec"] = float(m.group(3))
            if not skip_ttl:
                self._latest["tok_s_time"] = time.time()
            self._latest["has_timing"] = True
            self._latest["last_update"] = time.time()
            return

        # draft acceptance = 0.82738 (139 accepted / 168 generated)
        m = re.search(r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", line)
        if m:
            self._latest["draft_acceptance"] = float(m.group(1))
            self._latest["draft_accepted"] = int(m.group(2))
            self._latest["draft_generated"] = int(m.group(3))
            self._latest["draft_rate"] = float(m.group(1)) * 100  # as percentage
            self._latest["last_update"] = time.time()

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
            
            # Remove internal tracking fields from output
            result.pop("tok_s_time", None)
            result.pop("p_s_time", None)
            
            return result


# Global log parser instance
log_parser = LmStudioLogParser()


# ─── LM Studio stats tracker ───────────────────────────────────────
class LMStudioTracker:
    """Track active LM Studio sessions with accurate token stats."""

    def __init__(self):
        self.sessions: dict = {}
        self._lock = asyncio.Lock()
        # Cache for is_running check
        self._running_cached = False
        self._running_check_time = 0
        # Queue tracking
        self._queue_count = 0
        # Token counters (split by input/output)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        # Last known model name
        self._last_model = ""

    async def get_loaded_models(self) -> list:
        """Detect which models are loaded in VRAM by checking GPU memory + session history."""
        loaded = []
        try:
            # Get recent active models from sessions (last 10 minutes)
            async with self._lock:
                recent = {}
                now = time.time()
                for sid, s in self.sessions.items():
                    if now - s["start_time"] < 600:
                        recent[s["model"]] = now - s["start_time"]

            # Check GPU VRAM usage to determine how many models are loaded
            result = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            loaded_gpus = 0
            for line in stdout.decode().strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    mem = int(parts[0])
                    # GPU with >2GB VRAM likely has a model loaded
                    if mem > 2048:
                        loaded_gpus += 1

            # Return recent models up to the number of loaded GPUs
            if recent and loaded_gpus > 0:
                # Sort by most recent first
                sorted_models = sorted(recent.items(), key=lambda x: x[1])
                loaded = [m for m, _ in sorted_models[:loaded_gpus]]
            elif loaded_gpus > 0:
                # No recent sessions, get models from API
                if DASHBOARD_MODE == "ollama":
                    # Ollama: use /api/tags endpoint
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{LM_STUDIO_URL}/api/tags",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                loaded = [m["name"] for m in data.get("models", [])[:loaded_gpus]]
                else:
                    # LM Studio: use /v1/models endpoint
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{LM_STUDIO_URL}/v1/models",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                loaded = [m["id"] for m in data.get("data", [])[:loaded_gpus]]
        except Exception:
            pass
        return loaded

    async def get_models(self) -> list:
        """Get list of available models from LM Studio or Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                if DASHBOARD_MODE == "ollama":
                    url = f"{LM_STUDIO_URL}/api/tags"
                else:
                    url = f"{LM_STUDIO_URL}/v1/models"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if DASHBOARD_MODE == "ollama":
                            return [m["name"] for m in data.get("models", [])]
                        else:
                            return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    async def is_running(self) -> bool:
        """Check if LM Studio is running (cached)."""
        now = time.time()
        # Cache for 5 seconds
        if now - self._running_check_time < 5:
            return self._running_cached
        self._running_cached = False
        self._running_check_time = now
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{LM_STUDIO_URL}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    self._running_cached = resp.status == 200
                    return self._running_cached
        except Exception:
            return False

    async def start_session(self, request_json: dict) -> str:
        """Register a new session. Returns session_id."""
        session_id = f"sess_{int(time.time()*1000) % 1000000}"
        start_time = time.time()
        model = request_json.get("model", "unknown")
        async with self._lock:
            self._queue_count += 1  # Increment queue when request starts
            self._last_model = model  # Store last known model
            self.sessions[session_id] = {
                "session_id": session_id,
                "model": model,
                "start_time": start_time,
                "first_token_time": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tokens_per_sec": 0,
                "prompt_tokens_per_sec": 0,
                "draft_total": 0,
                "draft_accepted": 0,
                "draft_rejected": 0,
                "draft_ignored": 0,
                "draft_accept_rate": 0,
                "status": "prompting",
                "prompt_progress": 0,
                "last_update": start_time,
                "n_token": 0,
                # Token rate tracking
                "_token_count": 0,
                "_token_times": [],
                # Rate tracking
                "_content_samples": [],  # (timestamp, cumulative_content_chars)
                "_total_content": 0,
            }
        return session_id

    async def update_session(self, session_id: str, chunk_data: dict):
        """Update session stats from a streaming chunk."""
        async with self._lock:
            if session_id not in self.sessions:
                return
            s = self.sessions[session_id]
            now = time.time()

            # Parse choices/delta
            for choice in chunk_data.get("choices", []):
                delta = choice.get("delta", {})
                content = (delta.get("content") or "")
                reasoning = (delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason")

                # Track first token arrival (content or reasoning)
                if (content or reasoning) and s["first_token_time"] is None:
                    s["first_token_time"] = now
                    s["status"] = "generating"
                    self._queue_count = max(0, self._queue_count - 1)  # Decrement queue when generation starts

                # Track token count for rate calculation
                if content or reasoning:
                    s["_token_count"] = s.get("_token_count", 0) + 1
                    s["_token_times"].append(now)
                    # Keep only last 10 samples for rolling average
                    if len(s["_token_times"]) > 10:
                        s["_token_times"] = s["_token_times"][-10:]

                if finish:
                    s["status"] = "completed"
                    s["prompt_progress"] = 100

            # Parse usage (final chunk has authoritative counts)
            usage = chunk_data.get("usage", {})
            if usage:
                s["prompt_tokens"] = usage.get("prompt_tokens", s["prompt_tokens"])
                s["completion_tokens"] = usage.get("completion_tokens", s["completion_tokens"])
                s["total_tokens"] = usage.get("total_tokens", s["total_tokens"])
                # Increment global token counters (split by input/output)
                self._total_input_tokens += s["prompt_tokens"]
                self._total_output_tokens += s["completion_tokens"]
                self._session_input_tokens += s["prompt_tokens"]
                self._session_output_tokens += s["completion_tokens"]

            # Parse stats (MTP/draft token info)
            stats = chunk_data.get("stats", {})
            if stats:
                s["draft_total"] = stats.get("total_draft_tokens_count", s["draft_total"])
                s["draft_accepted"] = stats.get("accepted_draft_tokens_count", s["draft_accepted"])
                s["draft_rejected"] = stats.get("rejected_draft_tokens_count", s["draft_rejected"])
                s["draft_ignored"] = stats.get("ignored_draft_tokens_count", s["draft_ignored"])
                if s["draft_total"] > 0:
                    s["draft_accept_rate"] = round(s["draft_accepted"] / s["draft_total"] * 100, 1)

            # Prompt progress estimation
            elapsed = now - s["start_time"]
            if s["status"] == "prompting":
                s["prompt_progress"] = min(95, int(elapsed * 40))
            elif s["status"] == "generating":
                s["prompt_progress"] = 100

            # Rate tracking: sample content length periodically
            if s["_total_content"] > 0 and (not s["_content_samples"] or now - s["_content_samples"][-1][0] >= 0.5):
                s["_content_samples"].append((now, s["_total_content"]))
                # Keep last 10 seconds
                cutoff = now - 10
                s["_content_samples"] = [(t, c) for t, c in s["_content_samples"] if t >= cutoff]

                # Calculate chars/sec rate from content samples
                if len(s["_content_samples"]) >= 2:
                    t1, c1 = s["_content_samples"][-2]
                    t2, c2 = s["_content_samples"][-1]
                    dt = t2 - t1
                    if dt > 0.1:
                        chars_per_sec = (c2 - c1) / dt
                        s["tokens_per_sec"] = round(chars_per_sec / 4, 1)

            # Calculate prompt processing rate
            if s["first_token_time"] and s["prompt_tokens"] > 0:
                prompt_dur = s["first_token_time"] - s["start_time"]
                if prompt_dur >= 0.01:
                    s["prompt_tokens_per_sec"] = round(s["prompt_tokens"] / prompt_dur, 1)
                else:
                    total_dur = now - s["start_time"]
                    s["prompt_tokens_per_sec"] = round(s["prompt_tokens"] / max(total_dur, 0.1), 1)

            s["last_update"] = now

    async def finish_session(self, session_id: str, usage: dict = None):
        """Finalize a session with usage data."""
        async with self._lock:
            if session_id not in self.sessions:
                return
            s = self.sessions[session_id]
            s["status"] = "completed"
            s["prompt_progress"] = 100
            # Only decrement queue if first token never arrived (failed/cancelled request)
            if s["first_token_time"] is None:
                self._queue_count = max(0, self._queue_count - 1)
            now = time.time()
            if usage:
                s["prompt_tokens"] = usage.get("prompt_tokens", s["prompt_tokens"])
                s["completion_tokens"] = usage.get("completion_tokens", s["completion_tokens"])
                s["total_tokens"] = usage.get("total_tokens", s["total_tokens"])
                s["draft_total"] = usage.get("total_draft_tokens_count", s["draft_total"])
                s["draft_accepted"] = usage.get("accepted_draft_tokens_count", s["draft_accepted"])
                s["draft_rejected"] = usage.get("rejected_draft_tokens_count", s["draft_rejected"])
                s["draft_ignored"] = usage.get("ignored_draft_tokens_count", s["draft_ignored"])
                if s["draft_total"] > 0:
                    s["draft_accept_rate"] = round(s["draft_accepted"] / s["draft_total"] * 100, 1)

            # Calculate final token rate
            if len(s["_content_samples"]) >= 2:
                t1, c1 = s["_content_samples"][0]
                t2, c2 = s["_content_samples"][-1]
                dt = t2 - t1
                if dt > 0.1:
                    s["tokens_per_sec"] = round((c2 - c1) / dt / 4, 1)
            elif s["_total_content"] > 0 and s["first_token_time"]:
                gen_dur = now - s["first_token_time"]
                if gen_dur > 0.1:
                    s["tokens_per_sec"] = round(s["_total_content"] / gen_dur / 4, 1)

            # Calculate prompt processing rate
            if s["first_token_time"] and s["prompt_tokens"] > 0:
                prompt_dur = s["first_token_time"] - s["start_time"]
                if prompt_dur >= 0.01:
                    s["prompt_tokens_per_sec"] = round(s["prompt_tokens"] / prompt_dur, 1)
                else:
                    total_dur = now - s["start_time"]
                    s["prompt_tokens_per_sec"] = round(s["prompt_tokens"] / max(total_dur, 0.1), 1)

    async def get_active_stats(self) -> list:
        """Get stats for active/recent sessions."""
        async with self._lock:
            now = time.time()
            result = []
            for sid, s in list(self.sessions.items()):
                # Remove sessions older than 2 minutes that are completed
                if now - s["start_time"] > 120 and s["status"] == "completed":
                    continue
                r = dict(s)
                # Calculate tok/s from token times (rolling average)
                times = r.get("_token_times", [])
                if len(times) >= 2:
                    dt = times[-1] - times[0]
                    if dt > 0:
                        r["tokens_per_sec"] = round(len(times) / dt, 1)
                # Clean up internal fields
                r.pop("_content_samples", None)
                r.pop("_total_content", None)
                r.pop("_token_times", None)
                r.pop("_token_count", None)
                result.append(r)
            result.sort(key=lambda x: x["last_update"], reverse=True)
            return result

    def get_queue_length(self) -> int:
        """Get the number of requests waiting in queue."""
        return max(0, self._queue_count)

    def get_total_input_tokens(self) -> int:
        return self._total_input_tokens

    def get_total_output_tokens(self) -> int:
        return self._total_output_tokens

    def get_session_input_tokens(self) -> int:
        return self._session_input_tokens

    def get_session_output_tokens(self) -> int:
        return self._session_output_tokens

    def get_last_model(self) -> str:
        """Get the last known model name."""
        return self._last_model

    def reset_session_tokens(self) -> dict:
        """Reset session token counter and return previous values."""
        prev = {
            "input": self._session_input_tokens,
            "output": self._session_output_tokens,
        }
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        return prev


tracker = LMStudioTracker()


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
        # Get GPU count from nvidia-smi directly (avoid recursion)
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--list-gpus",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        gpu_count = len([l for l in stdout.decode().strip().split("\n") if l.strip()])
        if gpu_count == 0:
            return {}

        # Build PCI bus mapping: LACT GPU ID -> PCI bus -> nvidia-smi index
        # Get nvidia-smi PCI bus IDs
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
                    idx = int(parts[0])
                    pci = parts[1].strip()
                    nvidia_pci[pci] = idx

        # Query LACT for each GPU and match by PCI bus
        for lact_id in range(1, gpu_count + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "lact", "cli", "-g", str(lact_id), "stats",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                text = stdout.decode()

                # Extract PCI bus from first line: GPU 10DE:2204-1043:87B5-0000:03:00.0:
                pci_match = re.search(r"0000:(\w+):(\d+)\.(\d+)", text)
                if not pci_match:
                    continue
                pci_bus = f"0000:{pci_match.group(1)}:{pci_match.group(2)}.{pci_match.group(3)}"

                # Find matching nvidia-smi index (normalize PCI format)
                nvidia_idx = None
                for pci, idx in nvidia_pci.items():
                    # Normalize: extract bus:device.function and compare (case-insensitive)
                    # nvidia-smi: 00000000:03:00.0
                    # LACT: 0000:03:00.0
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
    """Reset the session token counter."""
    prev = tracker.reset_session_tokens()
    return {"success": True, "previous_tokens": prev}


# ─── OpenAI-compatible API ─────────────────────────────────────────
@app.get("/v1/models")
async def v1_models():
    models = await tracker.get_models()
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "lm-studio"} for m in models],
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    """OpenAI-compatible chat completions with accurate token tracking."""
    request_json = await request.json()
    is_stream = request_json.get("stream", False)
    session_id = await tracker.start_session(request_json)

    if is_stream:
        async def event_stream():
            try:
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.post(
                        f"{LM_STUDIO_URL}/v1/chat/completions",
                        json=request_json,
                        timeout=aiohttp.ClientTimeout(total=300, connect=10),
                    ) as resp:
                        async for line in resp.content:
                            line_str = line.decode("utf-8", errors="replace")
                            if line_str.startswith("data: "):
                                data_str = line_str[6:].strip()
                                if data_str and data_str != "[DONE]":
                                    try:
                                        chunk = json.loads(data_str)
                                        await tracker.update_session(session_id, chunk)
                                    except json.JSONDecodeError:
                                        pass
                            yield line_str
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'
            finally:
                await tracker.finish_session(session_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id,
            },
        )
    else:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json=request_json,
                timeout=aiohttp.ClientTimeout(total=300, connect=10),
            ) as resp:
                body = await resp.read()
                try:
                    rj = json.loads(body)
                    usage = rj.get("usage", {})
                    stats = rj.get("stats", {})
                    await tracker.finish_session(session_id, {**usage, **stats})
                except (json.JSONDecodeError, KeyError):
                    pass
                return Response(
                    content=body,
                    media_type="application/json",
                    status_code=resp.status,
                )


@app.post("/api/lmstudio/proxy")
async def lm_proxy(request: Request):
    return await v1_chat_completions(request)


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


async def _parse_ollama_logs() -> dict:
    """Parse Ollama logs from journalctl for timing stats."""
    result = {
        "prompt_progress": 0,
        "tokens_per_sec": 0,
        "prompt_tokens_per_sec": 0,
        "n_decoded": 0,
        "n_prompt": 0,
        "model": "",
        "has_timing": False,
        "last_update": 0,
        "draft_acceptance_rate": 0,
        "draft_tokens": 0,
    }
    try:
        # Get recent logs (last 30 seconds)
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "-u", "ollama.service", "--no-pager",
            "--since", "30 seconds ago",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode().split("\n")

        for line in lines:
            # Prompt processing: prompt processing, n_tokens = 4429, progress = 1.00, t = 8.71 s / 508.58 tokens per second
            m = re.search(r"prompt processing.*n_tokens\s*=\s*(\d+).*t\s*=\s*[\d.]+\s*s\s*/\s*([\d.]+)\s*tokens per second", line)
            if m:
                result["n_prompt"] = int(m.group(1))
                result["prompt_tokens_per_sec"] = float(m.group(2))
                result["has_timing"] = True
                result["last_update"] = time.time()

            # Generation: n_decoded = 101, tg = 134.21 t/s
            m = re.search(r"n_decoded\s*=\s*(\d+).*tg\s*=\s*([\d.]+)\s*t/s", line)
            if m:
                result["n_decoded"] = int(m.group(1))
                result["tokens_per_sec"] = float(m.group(2))
                result["has_timing"] = True
                result["last_update"] = time.time()

            # Progress: progress = 0.85
            m = re.search(r"progress\s*=\s*([\d.]+)", line)
            if m:
                result["prompt_progress"] = float(m.group(1)) * 100

    except Exception:
        pass

    return result


async def _collect_stats():
    """Collect all stats in one go."""
    gpus = await get_gpu_stats()
    cpu = await get_cpu_stats()
    mem = await get_memory_stats()
    disk = await get_disk_stats()

    lm_running = await tracker.is_running()
    lm_sessions = await tracker.get_active_stats() if lm_running else []
    lm_loaded = await tracker.get_loaded_models() if lm_running else []

    # Log parser only for LM Studio mode
    if DASHBOARD_MODE == "lmstudio":
        # Ensure log parser is running
        if not log_parser._watch_task or log_parser._watch_task.done():
            await log_parser.start_watching()
        lm_log = await log_parser.get_latest()
    else:
        # Ollama mode: parse logs from journalctl
        lm_log = await _parse_ollama_logs()

    # Merge session-based rates with log parser values
    # Session tracker has real-time tok/s from content samples
    # Use log parser values directly (no session override)
    for sess in lm_sessions:
        status = sess.get("status", "")
        sess_progress = sess.get("prompt_progress", 0) or 0

        # Update progress from session (only for active sessions)
        if status in ("prompting", "generating") and sess_progress > 0:
            lm_log["prompt_progress"] = sess_progress

        # Use session model name if log parser didn't find it
        if not lm_log.get("model") and sess.get("model"):
            lm_log["model"] = sess["model"]

    # Use last known model name if still empty
    if not lm_log.get("model"):
        lm_log["model"] = tracker.get_last_model()

    # Use loaded models from LM Studio API if still empty
    if not lm_log.get("model") and lm_loaded:
        lm_log["model"] = lm_loaded[0]
    
    # Calculate total power (sum of all GPU power draws)
    total_power = sum(g.get("power_draw", 0) for g in gpus if isinstance(g, dict))

    # Calculate costs
    total_input = tracker.get_total_input_tokens()
    total_output = tracker.get_total_output_tokens()
    session_input = tracker.get_session_input_tokens()
    session_output = tracker.get_session_output_tokens()
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
            "running": lm_running,
            "loaded_models": lm_loaded,
            "active_sessions": lm_sessions,
            "log_stats": lm_log,
            "queue_length": tracker.get_queue_length(),
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
    uvicorn.run("main:app", host=DASHBOARD_HOST, port=DASHBOARD_PORT, log_level="info", timeout_graceful_shutdown=2)
