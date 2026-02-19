#!/usr/bin/env python3
import argparse
import errno
import os
import signal
import subprocess
import sys
import re
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional

_TIME_PREFIX_RE = re.compile(r"^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]\s*")

def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if pid == os.getpid():
        return True

    try:
        os.kill(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        return False

    return True


def _terminate_pid(pid: int, purpose: str) -> None:
    if not _pid_is_running(pid):
        return

    print(f"[{_timestamp()}] Terminating stale {purpose} process {pid}.")
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return

    timeout = time.time() + 2.5
    while time.time() < timeout:
        if not _pid_is_running(pid):
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def _clean_child_line(line: str) -> str:
    line = line.rstrip("\n").rstrip("\r")
    if not line:
        return ""

    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    # Remove wrappers and timestamps repeatedly because sidecar output may include
    # chained prefixes such as "[time] [streamdiffusion][sidecar][stderr]".
    for _ in range(6):
        prev = line
        line = re.sub(
            r"^\[(\d{4}-\d{2}-\d{2}\s+)(\d{2}:\d{2}:\d{2})\]",
            "",
            line,
        )
        line = re.sub(
            r"^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]\s*",
            "",
            line,
        )
        line = re.sub(
            r"^\[streamdiffusion\](?:\[[^\]]+\])*\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"^\[(?:stdout|stderr)\](?:\[[^\]]+\])*\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )

        if line == prev:
            break

    line = _TIME_PREFIX_RE.sub("", line)
    return line.strip()


def _heartbeat_age_seconds(path: Path) -> Optional[float]:
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return None


def _read_pipe(stream) -> None:
    for line in iter(stream.readline, ""):
        if not line:
            break
        text = _clean_child_line(line)
        if not text:
            continue

        print(text)
        sys.stdout.flush()


def _get_signal_group_kill_funcs(process: subprocess.Popen):
    def _send(signal_name: int) -> None:
        if process.poll() is not None:
            return

        if os.name == "nt":
            try:
                process.send_signal(signal_name)
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass
            return

        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal_name)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

    def _term() -> None:
        _send(signal.SIGTERM)

    def _kill() -> None:
        _send(signal.SIGKILL)

    return _term, _kill


class BackendLifecycleManager:
    def __init__(
        self,
        command: list[str],
        workdir: str | None = None,
        pid_file: str | None = None,
        max_restarts: int = 2,
        restart_delay_seconds: float = 2.0,
        graceful_shutdown_seconds: float = 10.0,
        parent_pid: int | None = None,
        parent_check_interval_seconds: float = 2.0,
        heartbeat_file: str | None = None,
        heartbeat_timeout_seconds: float = 0.0,
        heartbeat_interval_seconds: float = 2.0,
    ):
        self.command = command
        self.workdir = workdir
        self.pid_file = Path(pid_file) if pid_file else None
        self.max_restarts = max_restarts
        self.restart_delay_seconds = restart_delay_seconds
        self.graceful_shutdown_seconds = graceful_shutdown_seconds
        self.parent_pid = parent_pid
        self.parent_check_interval_seconds = parent_check_interval_seconds
        self.heartbeat_timeout_seconds = max(0.0, heartbeat_timeout_seconds)
        self.heartbeat_interval_seconds = max(0.1, heartbeat_interval_seconds)

        self._process: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._restarts_used = 0
        self._stopping = threading.Event()
        self._monitor_stop = threading.Event()
        self._heartbeat_monitor_stop = threading.Event()
        self._parent_monitor: threading.Thread | None = None
        self._heartbeat_started_at: float = 0.0
        self._heartbeat_file = Path(heartbeat_file) if heartbeat_file else None
        self._heartbeat_monitor: threading.Thread | None = None

    def start(self) -> int:
        self._install_signal_handlers()
        self._register_pid_file()

        last_return_code: int = 0
        run_count = 0
        try:
            while True:
                if self._stopping.is_set():
                    return 0

                run_count += 1
                print(
                    f"[{_timestamp()}] Lifecycle start #{run_count} -> {self.command[0]}"
                )
                last_return_code = self._start_once_and_wait()

                if self._stopping.is_set():
                    return last_return_code

                if last_return_code == 0:
                    return 0

                if self._restarts_used >= self.max_restarts:
                    print(
                        f"[{_timestamp()}] Backend exited with code {last_return_code} and "
                        f"restart limit reached ({self._restarts_used}/{self.max_restarts})."
                    )
                    return last_return_code

                self._restarts_used += 1
                print(
                    f"[{_timestamp()}] Backend exited with code {last_return_code}; "
                    f"auto-restart {self._restarts_used}/{self.max_restarts} after "
                    f"{self.restart_delay_seconds:.1f}s."
                )
                time.sleep(self.restart_delay_seconds)
        except KeyboardInterrupt:
            print(f"[{_timestamp()}] KeyboardInterrupt received, stopping lifecycle.")
            self.stop()
            return 130
        finally:
            self._cleanup_process()
            self._stop_heartbeat_monitor()
            self._clear_heartbeat_file()
            self._unregister_pid_file()

    def stop(self) -> None:
        self._stopping.set()
        self._clear_heartbeat_file()
        self._cleanup_process()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _register_pid_file(self) -> None:
        if self.pid_file is None:
            return

        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        previous_pid = None
        if self.pid_file.exists():
            try:
                content = self.pid_file.read_text(encoding="utf-8").strip()
                if content:
                    previous_pid = int(content)
            except Exception:
                previous_pid = None

        if previous_pid is not None and previous_pid != os.getpid():
            _terminate_pid(previous_pid, "previous sidecar")

        self.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    def _unregister_pid_file(self) -> None:
        if self.pid_file is None:
            return

        try:
            if self.pid_file.exists() and self.pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.pid_file.unlink()
        except Exception:
            pass

    def _clear_heartbeat_file(self) -> None:
        if self._heartbeat_file is None:
            return

        try:
            self._heartbeat_file.unlink()
        except Exception:
            pass

    def _start_heartbeat_monitor(self) -> None:
        if self._heartbeat_file is None:
            return
        if self.heartbeat_timeout_seconds <= 0:
            print(f"[{_timestamp()}] Heartbeat monitor disabled.")
            return

        self._stop_heartbeat_monitor()
        self._heartbeat_monitor_stop.clear()
        self._heartbeat_started_at = time.time()
        self._heartbeat_monitor = threading.Thread(target=self._heartbeat_watch_loop, daemon=True)
        self._heartbeat_monitor.start()

    def _stop_heartbeat_monitor(self) -> None:
        thread = self._heartbeat_monitor
        self._heartbeat_monitor = None
        self._heartbeat_monitor_stop.set()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=1.5)

    def _heartbeat_watch_loop(self) -> None:
        while not self._heartbeat_monitor_stop.is_set() and not self._stopping.is_set():
            age = _heartbeat_age_seconds(self._heartbeat_file) if self._heartbeat_file else None
            if age is None:
                elapsed = time.time() - self._heartbeat_started_at
                if elapsed > self.heartbeat_timeout_seconds * 1.8:
                    print(
                        f"[{_timestamp()}] Backend heartbeat lease missing for too long; stopping."
                    )
                    self.stop()
                    return
            elif age > self.heartbeat_timeout_seconds:
                print(
                    f"[{_timestamp()}] Backend heartbeat lease stale ({round(age, 1)}s); stopping."
                )
                self.stop()
                return

            if self._heartbeat_monitor_stop.wait(self.heartbeat_interval_seconds):
                return

    def _start_once_and_wait(self) -> int:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        command = list(self.command)
        if not command:
            raise ValueError("No command provided to launch.")

        start_new_session = os.name != "nt"
        process = subprocess.Popen(
            command,
            cwd=self.workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=start_new_session,
        )
        self._process = process

        self._stdout_thread = threading.Thread(
            target=_read_pipe, args=(process.stdout,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=_read_pipe, args=(process.stderr,), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._start_parent_monitor()
        self._start_heartbeat_monitor()

        term_group, kill_group = _get_signal_group_kill_funcs(process)
        return_code: int | None = None

        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            print(f"[{_timestamp()}] KeyboardInterrupt, terminating backend process...")
            term_group()
            return_code = self._wait_with_timeout(process, self.graceful_shutdown_seconds)
            if return_code is None:
                print(
                    f"[{_timestamp()}] Backend did not stop, force killing..."
                )
                kill_group()
                return_code = self._wait_with_timeout(process, 3.0)
                if return_code is None:
                    return_code = 1
        finally:
            self._stop_parent_monitor()
            self._stop_heartbeat_monitor()
            if process.poll() is None:
                term_group()
                return_code = self._wait_with_timeout(process, self.graceful_shutdown_seconds)
            if process.poll() is None:
                kill_group()
                return_code = self._wait_with_timeout(process, 1.0)

            self._join_reader_threads()
            self._process = None

        code = int(return_code) if return_code is not None else 1
        print(f"[{_timestamp()}] Backend exited with code {code}")
        return code

    def _install_signal_handlers(self) -> None:
        def _on_signal(signum, frame) -> None:
            print(f"[{_timestamp()}] Received signal {signum}, stopping backend...")
            self.stop()

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
            try:
                signal.signal(sig, _on_signal)
            except Exception:
                pass

    def _cleanup_process(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return

        term_group, kill_group = _get_signal_group_kill_funcs(process)
        try:
            print(f"[{_timestamp()}] Stopping backend process for lifecycle shutdown.")
            term_group()
            terminated = self._wait_with_timeout(process, self.graceful_shutdown_seconds)
            if terminated is None:
                print(f"[{_timestamp()}] Force killing backend process...")
                kill_group()
                self._wait_with_timeout(process, 2.0)
        finally:
            self._stop_parent_monitor()
            self._join_reader_threads()
            if process.poll() is not None:
                print(f"[{_timestamp()}] Backend stopped with code {process.returncode}")
            self._process = None

    def _start_parent_monitor(self) -> None:
        if self.parent_pid is None:
            return

        self._stop_parent_monitor()
        self._monitor_stop.clear()

        def _watch():
            while not self._monitor_stop.is_set() and not self._stopping.is_set():
                if not _pid_is_running(self.parent_pid):
                    print(
                        f"[{_timestamp()}] Detected parent process {self.parent_pid} exited, "
                        "terminating backend lifecycle."
                    )
                    self.stop()
                    return

                if not self._monitor_stop.wait(self.parent_check_interval_seconds):
                    continue

        thread = threading.Thread(target=_watch, daemon=True)
        self._parent_monitor = thread
        thread.start()

    def _stop_parent_monitor(self) -> None:
        self._monitor_stop.set()
        thread = self._parent_monitor
        self._parent_monitor = None
        if thread is None:
            return
        if thread is threading.current_thread():
            return
        thread.join(timeout=1.5)

    def _join_reader_threads(self) -> None:
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._stdout_thread = None
        self._stderr_thread = None

    @staticmethod
    def _wait_with_timeout(process: subprocess.Popen, timeout_seconds: float) -> int | None:
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return None


def _launch_and_forward(
    argv: Iterable[str], workdir: str | None = None
) -> int:
    # Backward-compatible helper.
    manager = BackendLifecycleManager(command=list(argv), workdir=workdir)
    return manager.start()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and prefix stdout/stderr lines for backend logging"
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to run the command from",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=2,
        help="Maximum number of backend restarts after a non-zero exit before giving up.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=2.0,
        help="Delay in seconds before restarting backend after unexpected exit.",
    )
    parser.add_argument(
        "--graceful-shutdown-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for graceful shutdown before force-killing backend.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="Parent process ID. When absent, lifecycle shutdown when this process exits.",
    )
    parser.add_argument(
        "--parent-check-interval",
        type=float,
        default=2.0,
        help="Delay between parent liveness checks.",
    )
    parser.add_argument(
        "--heartbeat-file",
        default=None,
        help="Path to shared heartbeat lease file.",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=0.0,
        help="Maximum age (seconds) allowed for heartbeat lease before shutdown (<=0 disables watchdog).",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=2.0,
        help="Frequency in seconds for heartbeat lease checks.",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        help="Path to sidecar PID file.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --, e.g. -- python main.py --port 7860",
    )

    args = parser.parse_args()

    command = args.command
    if not command:
        parser.error("Missing command. Use -- <command> to pass the backend command.")
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Command cannot be empty after '--'.")

    manager = BackendLifecycleManager(
        command=command,
        workdir=args.workdir,
        max_restarts=args.max_restarts,
        restart_delay_seconds=args.restart_delay,
        graceful_shutdown_seconds=args.graceful_shutdown_timeout,
        parent_pid=args.parent_pid,
        parent_check_interval_seconds=args.parent_check_interval,
        pid_file=args.pid_file,
        heartbeat_file=args.heartbeat_file,
        heartbeat_timeout_seconds=args.heartbeat_timeout,
        heartbeat_interval_seconds=args.heartbeat_interval,
    )
    return manager.start()


if __name__ == "__main__":
    sys.exit(main())
