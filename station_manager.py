from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import subprocess
import sys
import os
import time
import signal
import threading
import psutil
import socket

app = Flask(__name__)
# Enable CORS for all domains
CORS(app)

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- LOGGING SERVICE ---
logs = []
MAX_LOGS = 2000

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "service": "station-manager",
        "backend_status": get_backend_status()
    })

@app.route('/log', methods=['POST'])
def add_log():
    try:
        entry = request.json
        if not entry:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.now().isoformat()
            
        logs.append(entry)
        if len(logs) > MAX_LOGS:
            logs.pop(0)
            
        return jsonify({"status": "logged", "count": len(logs)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/log', methods=['GET'])
def get_logs():
    return jsonify(logs)

@app.route('/log/clear', methods=['POST'])
def clear_logs():
    global logs
    logs = []
    return jsonify({"status": "cleared"}), 200

# --- PROCESS MANAGEMENT ---
BACKEND_PROCESS = None
BACKEND_SCRIPT = "start_server.py"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_process_on_port(port):
    """Find and kill any process listening on the specified port"""
    killed_count = 0
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        print(f"[Manager] Killing external process {proc.pid} on port {port}")
                        proc.kill()
                        killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"[Manager] Error killing process on port {port}: {e}")
    return killed_count > 0

def get_backend_status():
    global BACKEND_PROCESS
    
    # 1. Check if our managed process is running
    internal_running = False
    if BACKEND_PROCESS is not None and BACKEND_PROCESS.poll() is None:
        internal_running = True
        
    # 2. Check if the port is actually in use (External/Manual process)
    port_active = is_port_in_use(2020)

    if internal_running:
        return "running"
    elif port_active:
        return "running (external)" # Indicate it's running but not by us
    elif BACKEND_PROCESS is not None:
        return "crashed"
    else:
        return "stopped"

def start_backend_process():
    global BACKEND_PROCESS
    
    current_status = get_backend_status()
    if "running" in current_status:
        return False, f"Already running ({current_status})"
    
    try:
        # Use the same python interpreter
        python_exe = sys.executable
        cwd = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(cwd, BACKEND_SCRIPT)
        
        # Spawn process
        BACKEND_PROCESS = subprocess.Popen(
            [python_exe, script_path],
            cwd=cwd
        )
        _log_internal("INFO", f"Backend process started (PID: {BACKEND_PROCESS.pid})")
        return True, "Started"
    except Exception as e:
        return False, str(e)

def stop_backend_process():
    global BACKEND_PROCESS
    msg = ""
    
    # 1. Stop Managed Process
    if BACKEND_PROCESS:
        try:
            BACKEND_PROCESS.terminate()
            try:
                BACKEND_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                BACKEND_PROCESS.kill()
            msg = "Stopped managed process"
        except Exception as e:
            msg = f"Error stopping managed process: {e}"
        BACKEND_PROCESS = None
    
    # 2. Force Cleanup of Port 2020 (Kill Zombies/External)
    if is_port_in_use(2020):
        if kill_process_on_port(2020):
            msg += " (Killed external process)"
        else:
            msg += " (Failed to kill external process)"
            
    if not msg:
        msg = "Already stopped"
    else:
        # Log successful stop
        _log_internal("INFO", f"Backend stopped: {msg}")
        
    return True, msg

def _log_internal(level, message):
    """Helper to log internal events to the in-memory store"""
    try:
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "source": "StationManager"
        })
        # Keep log size in check
        if len(logs) > MAX_LOGS:
            logs.pop(0)
    except:
        pass

@app.route('/control/status', methods=['GET'])
def control_status():
    status = get_backend_status()
    pid = None
    if BACKEND_PROCESS:
        pid = BACKEND_PROCESS.pid
    elif status == "running (external)":
        # Try to find external PID
        try:
            for proc in psutil.process_iter(['pid', 'connections']):
                 for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == 2020:
                        pid = proc.pid
                        break
        except: pass

    return jsonify({
        "status": status,
        "pid": pid
    })

@app.route('/control/start', methods=['POST'])
def control_start():
    success, msg = start_backend_process()
    if success:
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500

@app.route('/control/stop', methods=['POST'])
def control_stop():
    success, msg = stop_backend_process()
    if success:
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500

@app.route('/control/restart', methods=['POST'])
def control_restart():
    stop_backend_process()
    time.sleep(1) # Give it a moment to release ports
    success, msg = start_backend_process()
    if success:
        return jsonify({"status": "success", "message": "Restarted"})
    else:
        return jsonify({"status": "error", "message": msg}), 500

# --- MEMORY MONITORING ---
MONITOR_INTERVAL = 10
MEMORY_THRESHOLD_PERCENT = 90
monitor_thread = None

def monitor_memory_loop():
    print(f"[Monitor] Starting Memory Watchdog (Threshold: {MEMORY_THRESHOLD_PERCENT}%)")
    while True:
        try:
            time.sleep(MONITOR_INTERVAL)
            
            # Check if backend is running
            if BACKEND_PROCESS is None or BACKEND_PROCESS.poll() is not None:
                continue

            try:
                proc = psutil.Process(BACKEND_PROCESS.pid)
                mem_info = proc.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                
                # Check System Memory
                sys_mem = psutil.virtual_memory()
                percent_used = (mem_info.rss / sys_mem.total) * 100
                
                # Logic: If Single Process uses > 90% of Total System RAM
                if percent_used > MEMORY_THRESHOLD_PERCENT:
                    msg = f"[Monitor] CRITICAL: Backend using {percent_used:.1f}% of System RAM ({rss_mb:.0f}MB). Restarting..."
                    print(msg)
                    # Log internally
                    logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR", 
                        "message": msg,
                        "source": "StationManager"
                    })
                    
                    # Restart
                    stop_backend_process()
                    time.sleep(2)
                    start_backend_process()
                    
            except psutil.NoSuchProcess:
                pass # Process died naturally
            except Exception as e:
                print(f"[Monitor] Error checking memory: {e}")
                
        except Exception as e:
            print(f"[Monitor] Fatal Loop Error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    print("Starting Station Manager on port 3001...", flush=True)
    _log_internal("INFO", "Station Manager System Started")

    # Start Monitor Thread
    monitor_thread = threading.Thread(target=monitor_memory_loop, daemon=True)
    monitor_thread.start()
    
    # Auto-start backend on launch
    start_backend_process()
    
    try:
        app.run(host='0.0.0.0', port=3001, debug=False, use_reloader=False)
    finally:
        # Cleanup on exit
        stop_backend_process()
