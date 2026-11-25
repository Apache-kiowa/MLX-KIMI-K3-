"""Kimi K3.1 cluster panel. Runs on mac1 (rank 0), standard library only.

    /Users/Shared/k31/env/bin/python /Users/Shared/k31/kimi-k31-mlx/panel/k31_panel.py
    open http://localhost:7070

Every Mac: connection, memory, GPU, disk, weights part, wired limit, server process.
mac1 also: start/stop the model on all Macs, settings, logs, chat.
--host 0.0.0.0 opens it to the local network (there is no password: trusted networks only).
K31_PANEL_DEMO=1 shows made-up data (for trying the page without a cluster).
"""

import argparse
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DEMO = os.environ.get("K31_PANEL_DEMO") == "1"


# ---------------------------------------------------------------- config
def read_config():
    keys = "HOSTS BASE VENV DST HOSTFILE PORT TP K31_INDEXER_FP8 K31_PKG MLX_LM_DIR".split()
    script = f"source {shlex.quote(str(PKG / 'run' / 'config.sh'))}; " + "; ".join(
        f'echo "{k}=${{{k}}}"' for k in keys
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    cfg = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    cfg["HOSTS"] = cfg.get("HOSTS", "").split()
    return cfg


CFG = read_config()
BASE = Path(CFG.get("BASE") or "/Users/Shared/k31")
SETTINGS_FILE = BASE / "panel-settings.json"
SERVER_LOG = BASE / "server.log"
ACTION_LOG = BASE / "panel-actions.log"

DEFAULTS = {
    "port": int(CFG.get("PORT") or 8080),
    "temp": 0.6,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": 0.0,
    "max_tokens": 4096,
    "prefill_step_size": 512,
    "prompt_cache_size": 10,
    "decode_concurrency": 8,
    "prompt_concurrency": 2,
    "kv_bits": 0,
    "thinking": True,
    "indexer_fp8": CFG.get("K31_INDEXER_FP8", "0") == "1",
    "fast_synch": True,
    "log_level": "INFO",
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass
    return s


def save_settings(s):
    BASE.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def think_kwargs(on):
    # Kimi templates have used both names; a Jinja template ignores the one it doesn't know.
    return {"thinking": bool(on), "enable_thinking": bool(on)}


# ---------------------------------------------------------------- host stats
STATS_SH = r"""
DST=__DST__
echo "hostname=$(hostname -s)"
echo "os=$(sw_vers -productVersion 2>/dev/null)"
echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "memsize=$(sysctl -n hw.memsize)"
echo "wired_limit_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)"
echo "loadavg=$(sysctl -n vm.loadavg | tr -d '{}')"
echo "swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
echo "boottime=$(sysctl -n kern.boottime | sed 's/.*sec = \([0-9]*\).*/\1/')"
vm_stat | awk '
  /page size of/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) print "pagesize=" $i}
  /Pages free/ {gsub(/\./,"",$3); print "p_free=" $3}
  /Pages active/ {gsub(/\./,"",$3); print "p_active=" $3}
  /Pages inactive/ {gsub(/\./,"",$3); print "p_inactive=" $3}
  /Pages speculative/ {gsub(/\./,"",$3); print "p_spec=" $3}
  /Pages wired down/ {gsub(/\./,"",$4); print "p_wired=" $4}
  /Pages purgeable/ {gsub(/\./,"",$3); print "p_purge=" $3}
  /occupied by compressor/ {gsub(/\./,"",$5); print "p_comp=" $5}'
echo "gpu=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)"
df -k /Users/Shared | awk 'NR==2{printf "disk_total=%d\ndisk_avail=%d\n", $2*1024, $4*1024}'
ls -l "$DST"/tp*-r*-*.safetensors 2>/dev/null | awk '{n++; s+=$5} END{printf "w_files=%d\nw_bytes=%d\n", n, s}'
echo "w_tmp=$(ls -a "$DST" 2>/dev/null | grep -c '^\.tp')"
for f in "$DST"/tp*-r*.index.json; do [ -f "$f" ] && echo "w_index=$(basename "$f")" && grep -o '"total_size": *[0-9]*' "$f" | head -1 | sed 's/.*: */w_expect=/'; done
echo "server_procs=$(pgrep -f 'mlx_lm.server' | wc -l | tr -d ' ')"
echo "rdma=$(command -v rdma_ctl >/dev/null && rdma_ctl status 2>/dev/null | head -1)"
"""


def parse_stats(text):
    d = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()

    def num(k, default=0):
        try:
            return int(float(d.get(k) or default))
        except ValueError:
            return default

    page = num("pagesize", 16384)
    total = num("memsize")
    wired = num("p_wired") * page
    comp = num("p_comp") * page
    active = num("p_active") * page
    free = (num("p_free") + num("p_inactive") + num("p_spec") + num("p_purge")) * page
    used = min(total, wired + comp + active)
    return {
        "hostname": d.get("hostname", ""),
        "os": d.get("os", ""),
        "chip": d.get("chip", ""),
        "mem_total": total,
        "mem_used": used,
        "mem_wired": wired,
        "mem_comp": comp,
        "mem_free": free,
        "wired_limit_mb": num("wired_limit_mb"),
        "loadavg": d.get("loadavg", ""),
        "swap": d.get("swap", ""),
        "uptime_s": max(0, int(time.time()) - num("boottime", int(time.time()))),
        "gpu": num("gpu", -1) if d.get("gpu") else -1,
        "disk_total": num("disk_total"),
        "disk_avail": num("disk_avail"),
        "w_files": num("w_files"),
        "w_bytes": num("w_bytes"),
        "w_expect": num("w_expect"),
        "w_index": d.get("w_index", ""),
        "w_tmp": num("w_tmp"),
        "server_procs": num("server_procs"),
        "rdma": d.get("rdma", ""),
    }


def demo_stats(i, host):
    t = time.time()
    total = 512 * 2**30
    loaded = MODEL.state == "ready"
    wired = (402 if loaded else 3) * 2**30 + random.randint(0, 2**30)
    return {
        "hostname": host, "os": "26.2", "chip": "Apple M3 Ultra", "mem_total": total,
        "mem_used": wired + 9 * 2**30, "mem_wired": wired, "mem_comp": 0, "mem_free": total - wired,
        "wired_limit_mb": 480000 if i != 2 else 0, "loadavg": "1.2 1.0 0.9", "swap": "0.00M",
        "uptime_s": 86400 * 2 + i * 3600,
        "gpu": int(60 + 30 * abs(__import__("math").sin(t / 3 + i))) if loaded else random.randint(0, 3),
        "disk_total": 4 * 10**12 if i == 0 else 10**12, "disk_avail": 5 * 10**11,
        "w_files": 94 if i != 1 else 41, "w_bytes": 402_500_000_000 if i != 1 else 175_000_000_000,
        "w_expect": 402_500_000_000, "w_index": f"tp4-r{i}.index.json", "w_tmp": 1 if i == 1 else 0,
        "server_procs": 1 if loaded else 0, "rdma": "enabled",
    }


class Collector:
    def __init__(self):
        self.hosts = CFG["HOSTS"] or ["mac1"]
        self.data = {h: {"ok": False, "error": "ещё не опрошен", "ts": 0} for h in self.hosts}
        self.lock = threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def poll(self, i, host):
        t0 = time.time()
        try:
            if DEMO:
                time.sleep(0.02 + 0.01 * i)
                st = demo_stats(i, host)
            else:
                script = STATS_SH.replace("__DST__", shlex.quote(CFG.get("DST", "")))
                if i == 0:
                    cmd = ["bash", "-s"]
                else:
                    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, "bash -s"]
                r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=8)
                if r.returncode != 0 and "memsize=" not in r.stdout:
                    raise RuntimeError((r.stderr or "нет ответа").strip().splitlines()[-1][:200])
                st = parse_stats(r.stdout)
            st.update(ok=True, error="", latency_ms=int((time.time() - t0) * 1000))
        except subprocess.TimeoutExpired:
            st = {"ok": False, "error": "таймаут ssh (8 с)"}
        except Exception as e:  # noqa: BLE001
            st = {"ok": False, "error": str(e)}
        st.update(host=host, rank=i, ts=time.time())
        with self.lock:
            self.data[host] = st

    def loop(self):
        while True:
            ths = [threading.Thread(target=self.poll, args=(i, h)) for i, h in enumerate(self.hosts)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            time.sleep(3)

    def snapshot(self):
        with self.lock:
            return [dict(self.data[h]) for h in self.hosts]


# ---------------------------------------------------------------- model process
def log_action(msg):
    BASE.mkdir(parents=True, exist_ok=True)
    with open(ACTION_LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


class Model:
    def __init__(self):
        self.proc = None
        self.state = "stopped"  # stopped | starting | ready | stopping | failed
        self.started = 0
        self.ready_at = 0
        self.detail = ""
        self.cmd = []
        threading.Thread(target=self.watch, daemon=True).start()

    def build_cmd(self, s):
        venv = CFG.get("VENV", "")
        dst = CFG.get("DST", "")
        env = [
            "--env", f"K31_INDEXER_FP8={1 if s['indexer_fp8'] else 0}",
            "--env", f"K31_PRESPLIT_DIR={dst}",
        ]
        if s.get("fast_synch"):
            env += ["--env", "MLX_METAL_FAST_SYNCH=1"]
        srv = [
            f"{venv}/bin/python", "-m", "mlx_lm.server", "--model", dst, "--trust-remote-code",
            "--host", "0.0.0.0", "--port", str(s["port"]),
            "--temp", str(s["temp"]), "--top-p", str(s["top_p"]), "--top-k", str(s["top_k"]),
            "--min-p", str(s["min_p"]), "--max-tokens", str(s["max_tokens"]),
            "--prefill-step-size", str(s["prefill_step_size"]),
            "--prompt-cache-size", str(s["prompt_cache_size"]),
            "--decode-concurrency", str(s["decode_concurrency"]),
            "--prompt-concurrency", str(s["prompt_concurrency"]),
            "--log-level", s["log_level"],
            "--chat-template-args", json.dumps(think_kwargs(s["thinking"])),
        ]
        if int(s.get("kv_bits") or 0):
            srv += ["--kv-bits", str(s["kv_bits"])]
        return [f"{venv}/bin/mlx.launch", "--backend", "jaccl", "--hostfile", CFG.get("HOSTFILE", ""),
                *env, "--", *srv]

    def start(self):
        if self.state in ("starting", "ready", "stopping"):
            return False, "модель уже запущена или останавливается"
        s = load_settings()
        if not DEMO and not Path(CFG.get("HOSTFILE", "")).exists():
            return False, f"нет hostfile {CFG.get('HOSTFILE')}: сначала «Создать hostfile» (04_cluster.sh)"
        self.cmd = self.build_cmd(s)
        log_action("START " + " ".join(shlex.quote(c) for c in self.cmd))
        logf = open(SERVER_LOG, "a")
        logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} запуск =====\n")
        logf.flush()
        if DEMO:
            cmd = ["bash", "-c", "for i in $(seq 1 6); do echo \"[demo] загрузка весов $((i*17))%\"; sleep 1; done; echo '[demo] server ready'; sleep 100000"]
        else:
            cmd = self.cmd
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=str(BASE))
        self.state, self.started, self.ready_at, self.detail = "starting", time.time(), 0, "загрузка весов"
        return True, "запуск"

    def stop(self):
        if self.proc is None and self.state == "stopped":
            return False, "модель не запущена"
        self.state, self.detail = "stopping", "остановка на всех маках"
        log_action("STOP")

        def _stop():
            if self.proc and self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(15)
                except Exception:  # noqa: BLE001
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
            if not DEMO:
                for i, h in enumerate(CFG["HOSTS"]):
                    c = "pkill -f 'mlx_lm.server' ; true"
                    cmd = ["bash", "-c", c] if i == 0 else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", h, c]
                    subprocess.run(cmd, capture_output=True, timeout=15)
            self.proc, self.state, self.detail = None, "stopped", ""

        threading.Thread(target=_stop, daemon=True).start()
        return True, "остановка"

    def healthy(self, port):
        if DEMO:
            return time.time() - self.started > 6
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def watch(self):
        while True:
            time.sleep(2)
            if self.state == "stopping":
                continue
            if self.proc is None:
                # A server started outside the panel (e.g. run/06_serve.sh) still counts.
                if DEMO:
                    continue
                up = self.healthy(load_settings()["port"])
                if up and self.state in ("stopped", "failed"):
                    self.state, self.ready_at, self.started = "ready", time.time(), time.time()
                    self.detail = "запущена вне панели (06_serve.sh)"
                    log_action("обнаружен сервер, запущенный вне панели")
                elif not up and self.state == "ready":
                    self.state, self.detail = "stopped", "сервер перестал отвечать"
                continue
            rc = self.proc.poll()
            if rc is not None:
                self.state, self.detail = "failed", f"процесс завершился с кодом {rc}: см. логи"
                log_action(f"EXIT {rc}")
                self.proc = None
                continue
            if self.state == "starting" and self.healthy(load_settings()["port"]):
                self.state, self.ready_at, self.detail = "ready", time.time(), ""
                log_action(f"READY за {int(self.ready_at - self.started)} с")

    def info(self):
        return {"state": self.state, "detail": self.detail, "started": self.started,
                "ready_at": self.ready_at, "cmd": " ".join(shlex.quote(c) for c in self.cmd)}


MODEL = Model()
COLLECTOR = None


# ---------------------------------------------------------------- actions
ACTIONS = {
    "hostfile": ("Создать hostfile и проверить версии (04_cluster.sh)", ["bash", str(PKG / "run" / "04_cluster.sh")]),
}
ACTION_STATE = {"running": None}


def run_action(name):
    if ACTION_STATE["running"]:
        return False, f"уже выполняется: {ACTION_STATE['running']}"
    title, cmd = ACTIONS[name]
    ACTION_STATE["running"] = title

    def _run():
        with open(ACTION_LOG, "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} >>> {title}\n")
            f.flush()
            if DEMO:
                f.write("[demo] ok\n")
            else:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            f.write(f"{time.strftime('%H:%M:%S')} <<< готово\n")
        ACTION_STATE["running"] = None

    threading.Thread(target=_run, daemon=True).start()
    return True, title


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        p = self.path.split("?")[0]
        q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) if "?" in self.path else {}
        if p in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/status":
            self.send_json({
                "hosts": COLLECTOR.snapshot(), "model": MODEL.info(), "demo": DEMO,
                "config": {k: CFG.get(k) for k in ("BASE", "DST", "HOSTFILE", "TP", "VENV")},
                "hostfile_exists": DEMO or Path(CFG.get("HOSTFILE", "")).exists(),
                "action": ACTION_STATE["running"], "now": time.time(),
            })
        elif p == "/api/settings":
            self.send_json({"settings": load_settings(), "defaults": DEFAULTS})
        elif p == "/api/logs":
            n = int(q.get("n", 400))
            self.send_json({"server": tail(SERVER_LOG, n), "actions": tail(ACTION_LOG, n)})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path
        if p == "/api/settings":
            s = load_settings()
            new = self.body()
            for k, v in new.items():
                if k in DEFAULTS:
                    s[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], bool) else bool(v)
            save_settings(s)
            self.send_json({"ok": True, "settings": s, "restart_needed": MODEL.state in ("starting", "ready")})
        elif p == "/api/model/start":
            ok, msg = MODEL.start()
            self.send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif p == "/api/model/stop":
            ok, msg = MODEL.stop()
            self.send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif p.startswith("/api/action/"):
            name = p.rsplit("/", 1)[1]
            if name not in ACTIONS:
                return self.send_json({"error": "unknown action"}, 404)
            ok, msg = run_action(name)
            self.send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif p == "/api/chat":
            self.chat(self.body())
        else:
            self.send_json({"error": "not found"}, 404)

    def chat(self, req):
        s = load_settings()
        if MODEL.state != "ready" and not DEMO:
            return self.send_json({"error": "модель не запущена"}, 503)
        payload = {
            "messages": req.get("messages", []),
            "stream": True,
            "temperature": float(req.get("temperature", s["temp"])),
            "top_p": float(req.get("top_p", s["top_p"])),
            "max_tokens": int(req.get("max_tokens", s["max_tokens"])),
            "chat_template_kwargs": think_kwargs(req.get("thinking", s["thinking"])),
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if DEMO:
            return self.demo_stream(payload)
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{s['port']}/v1/chat/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}),
                timeout=3600)
            for line in r:
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            msg = json.dumps({"error": str(e)}, ensure_ascii=False)
            try:
                self.wfile.write(f"data: {msg}\n\n".encode())
            except Exception:  # noqa: BLE001
                pass

    def demo_stream(self, payload):
        last = payload["messages"][-1]["content"] if payload["messages"] else ""
        parts = []
        if payload["chat_template_kwargs"]["thinking"]:
            parts += [("reasoning", w + " ") for w in "Пользователь пишет в демо-режиме. Отвечу коротко.".split()]
        parts += [("content", w + " ") for w in f"Это демо-ответ панели. Вы написали: «{last[:80]}». Настоящая модель ответит, когда кластер будет запущен.".split()]
        try:
            for key, w in parts:
                chunk = {"choices": [{"delta": {key: w}, "index": 0}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.04)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global COLLECTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = open to the local network (no password)")
    ap.add_argument("--port", type=int, default=7070)
    a = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    COLLECTOR = Collector()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    shown = socket.gethostname() if a.host == "0.0.0.0" else "localhost"
    print(f"Kimi K3.1 panel: http://{shown}:{a.port}  (hosts: {' '.join(COLLECTOR.hosts)}{', DEMO' if DEMO else ''})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
