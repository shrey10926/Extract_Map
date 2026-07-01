"""renesas_master_app.py — PyQt6 desktop UI for the Renesas master-clock time sync.

This is a thin, resilient GUI shell around renesas_master_backend.py. It does NOT
re-implement the wire protocol: it reuses the backend's verified build_payload() and
socket tuning, and simply drives them from a config file and surfaces live status.

Features
    * Enter the master clock IP / port at runtime (remembered between launches).
    * Start / Stop, with automatic reconnect handled by the streaming thread.
    * Stacked display of PC time, the time sent to the master, and the raw protocol
      frame — each in a large, distinctly-coloured section for easy comparison.
    * JSON config alongside the app for protocol + timing/resilience knobs.
    * Rotating file logging in ./logs (app.log plus a per-second time_compare.log that
      records PC vs sent vs protocol time) and defensive error handling.

Run:  python renesas_master_app.py
"""
from __future__ import annotations

import copy
import json
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import renesas_master_backend as backend

# ── Paths & app identity ─────────────────────────────────────────────
APP_NAME = "RenesasTimeSync"


def _base_dir() -> Path:
    """Folder used for config.json and logs — kept alongside the app for easy edits.

    As a script this is the directory containing this file (the project directory).
    When frozen with PyInstaller it is the folder holding the .exe (never the
    temporary _MEIPASS unpack dir, which is wiped on each launch).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


_BASE_DIR = _base_dir()
CONFIG_FILE = _BASE_DIR / "config.json"
LOG_DIR = _BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
TIME_LOG_FILE = LOG_DIR / "time_compare.log"

log = logging.getLogger("renesas_app")
# Dedicated logger for the per-second PC vs sent vs protocol time comparison.
time_log = logging.getLogger("renesas_app.timelog")

# Config seeded from the backend defaults so there is a single source of truth.
DEFAULT_CONFIG = {
    "network": {
        "device_ip": backend.DEVICE_IP,
        "device_port": backend.DEVICE_PORT,
        "source_id": backend.SOURCE_ID,
        "time_fmt": backend.TIME_FMT,
        "terminator": backend.TERMINATOR,
    },
    "timing": {
        "send_interval": backend.SEND_INTERVAL,
        "connect_timeout": backend.CONNECT_TIMEOUT,
        "send_timeout": backend.SEND_TIMEOUT,
        "reconnect_delay": backend.RECONNECT_DELAY,
        "dead_peer_timeout": backend.DEAD_PEER_TIMEOUT,
        "heartbeat_every": backend.HEARTBEAT_EVERY,
    },
}


# ── Config load / save ───────────────────────────────────────────────
def load_config() -> dict:
    """Load config from disk, merging over defaults. Creates the file on first run."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for section in ("network", "timing"):
                if isinstance(data.get(section), dict):
                    cfg[section].update(data[section])
            log.info("loaded config from %s", CONFIG_FILE)
        else:
            save_config(cfg)
            log.info("created default config at %s", CONFIG_FILE)
    except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError
        log.error("config load failed (%s); using defaults", exc)
    return cfg


def save_config(cfg: dict) -> None:
    """Persist config to disk (raises OSError on failure — caller decides severity)."""
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def apply_config_to_backend(cfg: dict) -> None:
    """Push config values into the backend module so its functions use them.

    The backend reads these as module-level names at call time, so setting them here
    reconfigures build_payload() and _configure_socket() without editing the backend.
    """
    net, tim = cfg["network"], cfg["timing"]
    backend.DEVICE_IP = str(net["device_ip"])
    backend.DEVICE_PORT = int(net["device_port"])
    backend.SOURCE_ID = str(net["source_id"])
    backend.TIME_FMT = str(net["time_fmt"])
    backend.TERMINATOR = str(net["terminator"])
    backend.SEND_INTERVAL = float(tim["send_interval"])
    backend.CONNECT_TIMEOUT = float(tim["connect_timeout"])
    backend.SEND_TIMEOUT = float(tim["send_timeout"])
    backend.RECONNECT_DELAY = float(tim["reconnect_delay"])
    backend.DEAD_PEER_TIMEOUT = int(tim["dead_peer_timeout"])
    backend.HEARTBEAT_EVERY = int(tim["heartbeat_every"])


# ── Logging (rotating file + console + in-app status bridge) ─────────
class LogBridge(QObject):
    """Carries log records from any thread to the GUI thread via a queued signal."""

    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.message.emit(self.format(record))
        except Exception:  # never let logging crash the app
            pass


def setup_logging(bridge: LogBridge) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(file_fmt)
    root.addHandler(console)

    qt_handler = QtLogHandler(bridge)
    qt_handler.setFormatter(logging.Formatter("%(levelname)s · %(message)s"))
    root.addHandler(qt_handler)

    # Per-second time comparison lives in its own file, kept out of app.log/console.
    time_handler = RotatingFileHandler(
        TIME_LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    time_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    time_log.setLevel(logging.INFO)
    time_log.addHandler(time_handler)
    time_log.propagate = False


# ── Streaming worker thread ──────────────────────────────────────────
class StreamThread(QThread):
    """Runs the connect/stream/reconnect loop off the UI thread.

    Emits `status(state, message)` for the indicator and
    `frame(sent_time, sent_date, protocol)` for each frame actually sent, and logs the
    per-second PC/sent/protocol comparison. Reuses the backend's build_payload().
    """

    status = pyqtSignal(str, str)       # connecting|connected|disconnected|error|stopped
    frame = pyqtSignal(str, str, str)   # (sent HH:MM:SS, sent dd:mm:yyyy, protocol frame)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # executes in the worker thread
        self._stop.clear()
        ip, port = backend.DEVICE_IP, backend.DEVICE_PORT
        while not self._stop.is_set():
            try:
                self.status.emit("connecting", f"Connecting to {ip}:{port} …")
                log.info("connecting to %s:%d", ip, port)
                with socket.create_connection(
                    (ip, port), timeout=backend.CONNECT_TIMEOUT
                ) as sock:
                    backend._configure_socket(sock)  # verified keepalive/timeout tuning
                    self.status.emit("connected", f"Connected to {ip}:{port}")
                    log.info("connected to %s:%d", ip, port)
                    self._send_loop(sock)
            except OSError as exc:
                if self._stop.is_set():
                    break
                delay = backend.RECONNECT_DELAY
                log.warning("connection lost (%s); reconnecting in %.0fs", exc, delay)
                self.status.emit(
                    "disconnected", f"Disconnected: {exc} — retrying in {delay:.0f}s"
                )
                self._stop.wait(delay)
            except Exception as exc:  # unexpected: log full trace, keep the app alive
                log.exception("unexpected error in stream thread")
                self.status.emit("error", f"Error: {exc}")
                self._stop.wait(backend.RECONNECT_DELAY)
        self.status.emit("stopped", "Stopped")
        log.info("stopped")

    def _send_loop(self, sock: socket.socket) -> None:
        # Align first send to the whole second, then hold a drift-free cadence.
        self._stop.wait(1.0 - (time.time() % 1.0))
        next_tick = time.monotonic()
        count = 0
        while not self._stop.is_set():
            now = datetime.now()
            payload = backend.build_payload(now)  # exact, verified wire frame
            sock.sendall(payload)
            count += 1
            pc_time = now.strftime("%H:%M:%S.%f")[:-3]   # PC time incl. milliseconds
            sent_time = now.strftime("%H:%M:%S")          # whole-second value sent
            protocol_text = payload.decode("ascii", "replace").replace("\r", "\\r")
            # Per-second three-way comparison, one line per send in its own log file.
            time_log.info(
                "PC=%s | SENT=%s | PROTOCOL=%s", pc_time, sent_time, protocol_text
            )
            self.frame.emit(sent_time, now.strftime("%d:%m:%Y"), protocol_text)
            if count % backend.HEARTBEAT_EVERY == 0:
                log.info("streaming — %d frames sent", count)
            next_tick += backend.SEND_INTERVAL
            self._stop.wait(max(0.0, next_tick - time.monotonic()))


# ── Main window ──────────────────────────────────────────────────────
STATE_COLORS = {
    "idle": "#9AA5B1",
    "connecting": "#F0A020",
    "connected": "#2E7D32",
    "disconnected": "#C0392B",
    "error": "#C0392B",
    "stopped": "#9AA5B1",
}

STYLESHEET = """
QWidget { background: #F4F6F9; color: #22303C; font-family: 'Segoe UI', sans-serif; font-size: 10.5pt; }
QLabel#title { font-size: 15pt; font-weight: 600; color: #1F2933; }
QLabel#statusLabel { color: #52606D; font-weight: 600; }
QGroupBox { border: 1px solid #E1E5EA; border-radius: 10px; margin-top: 14px; background: #FFFFFF; font-weight: 600; color: #52606D; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; }
QLineEdit { background: #FFFFFF; border: 1px solid #CBD2D9; border-radius: 8px; padding: 7px 10px; }
QLineEdit:focus { border: 1px solid #26890D; }
QLineEdit:disabled { background: #EEF1F4; color: #9AA5B1; }
QPushButton#start { background: #26890D; color: #FFFFFF; border: none; border-radius: 8px; padding: 9px 24px; font-weight: 600; }
QPushButton#start:hover { background: #1F6E0A; }
QPushButton#start:disabled { background: #BCC5CE; }
QPushButton#stop { background: #FFFFFF; color: #C0392B; border: 1px solid #E4A9A2; border-radius: 8px; padding: 9px 24px; font-weight: 600; }
QPushButton#stop:hover { background: #FCEBEA; }
QPushButton#stop:disabled { color: #BCC5CE; border-color: #E1E5EA; }
QFrame#card { background: #FFFFFF; border: 1px solid #E1E5EA; border-radius: 12px; }
QLabel#caption { color: #7B8794; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#cardSub { color: #7B8794; font-family: Consolas, 'Courier New', monospace; font-size: 13pt; }
QLabel#pcValue { color: #1F6FEB; font-family: Consolas, 'Courier New', monospace; font-size: 60px; font-weight: 700; }
QLabel#sentValue { color: #1E9E5A; font-family: Consolas, 'Courier New', monospace; font-size: 60px; font-weight: 700; }
QLabel#protoValue { color: #B45309; font-family: Consolas, 'Courier New', monospace; font-size: 30px; font-weight: 700; }
QStatusBar { background: #FFFFFF; border-top: 1px solid #E1E5EA; color: #52606D; }
QStatusBar::item { border: none; }
"""


class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.thread: StreamThread | None = None
        self._frames = 0

        self.setWindowTitle("Renesas Master Clock — Time Sync")
        self.setMinimumSize(660, 720)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._set_running(False)

        # Live PC clock, independent of the network.
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_pc_time)
        self._clock_timer.start(100)
        self._tick_pc_time()

    # -- UI construction ------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 12)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addWidget(self._build_connection_group())
        root.addLayout(self._build_sections(), 1)

        self.setCentralWidget(central)

        self._frames_label = QLabel("Frames: 0")
        self.statusBar().addPermanentWidget(self._frames_label)
        self.statusBar().showMessage("Idle")

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        title = QLabel("Renesas Master Clock — Time Sync")
        title.setObjectName("title")

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(14, 14)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusLabel")

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_dot)
        header.addSpacing(6)
        header.addWidget(self.status_label)
        return header

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Master Clock Connection")
        group.setToolTip(f"Advanced settings: {CONFIG_FILE}")
        row = QHBoxLayout(group)
        row.setContentsMargins(16, 8, 16, 14)
        row.setSpacing(10)

        self.ip_edit = QLineEdit(str(self.config["network"]["device_ip"]))
        self.ip_edit.setPlaceholderText("e.g. 192.168.1.111")
        self.ip_edit.setClearButtonEnabled(True)

        self.port_edit = QLineEdit(str(self.config["network"]["device_port"]))
        self.port_edit.setPlaceholderText("Port")
        self.port_edit.setValidator(QIntValidator(1, 65535, self))
        self.port_edit.setMaximumWidth(110)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self.on_start)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.clicked.connect(self.on_stop)

        row.addWidget(QLabel("Master IP"))
        row.addWidget(self.ip_edit, 2)
        row.addWidget(QLabel("Port"))
        row.addWidget(self.port_edit, 1)
        row.addStretch(1)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        return group

    def _make_card(self, caption: str, value_object: str, sub_object: str | None = None):
        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        box = QVBoxLayout(card)
        box.setContentsMargins(24, 14, 24, 14)
        box.setSpacing(4)

        cap = QLabel(caption)
        cap.setObjectName("caption")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)

        value = QLabel("—")
        value.setObjectName(value_object)
        value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value.setWordWrap(True)

        box.addStretch(1)
        box.addWidget(cap)
        box.addWidget(value)

        sub = None
        if sub_object is not None:
            sub = QLabel("")
            sub.setObjectName(sub_object)
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(sub)

        box.addStretch(1)
        return card, value, sub

    def _build_sections(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(14)

        pc_card, self.pc_time, self.pc_sub = self._make_card(
            "PC TIME  (SYSTEM CLOCK)", "pcValue", "cardSub"
        )
        sent_card, self.sent_time, self.sent_sub = self._make_card(
            "SENT TO MASTER  (TIME VALUE)", "sentValue", "cardSub"
        )
        proto_card, self.proto_value, _ = self._make_card(
            "PROTOCOL FRAME  (ON THE WIRE)", "protoValue"
        )

        for card in (pc_card, sent_card, proto_card):
            col.addWidget(card, 1)
        return col

    # -- Live PC clock --------------------------------------------------
    def _tick_pc_time(self) -> None:
        now = datetime.now()
        self.pc_time.setText(now.strftime("%H:%M:%S"))
        self.pc_sub.setText(now.strftime("%d:%m:%Y"))

    # -- Start / Stop ---------------------------------------------------
    def on_start(self) -> None:
        ip = self.ip_edit.text().strip()
        port_text = self.port_edit.text().strip()

        if not ip:
            QMessageBox.warning(self, "Missing IP", "Enter the master clock IP address.")
            return
        try:
            port = int(port_text)
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, "Invalid port", "Port must be a whole number between 1 and 65535."
            )
            return

        # Persist the entered address, then apply the whole config to the backend.
        self.config["network"]["device_ip"] = ip
        self.config["network"]["device_port"] = port
        try:
            save_config(self.config)
        except OSError as exc:  # non-fatal: run with current settings anyway
            log.error("could not save config (%s)", exc)
        apply_config_to_backend(self.config)

        self._frames = 0
        self._frames_label.setText("Frames: 0")

        self.thread = StreamThread()
        self.thread.status.connect(self.on_status)
        self.thread.frame.connect(self.on_frame)
        self.thread.finished.connect(self._on_thread_finished)
        self._set_running(True)
        self.thread.start()

    def on_stop(self) -> None:
        if self.thread and self.thread.isRunning():
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping …")
            self.statusBar().showMessage("Stopping …")
            self.thread.stop()

    # -- Signal handlers ------------------------------------------------
    def on_status(self, state: str, message: str) -> None:
        color = STATE_COLORS.get(state, STATE_COLORS["idle"])
        self.status_dot.setStyleSheet(f"background:{color}; border-radius:7px;")
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
        if state == "stopped":
            self._set_running(False)

    def on_frame(self, sent_time: str, sent_date: str, protocol_text: str) -> None:
        self.sent_time.setText(sent_time)
        self.sent_sub.setText(sent_date)
        self.proto_value.setText(protocol_text)
        self._frames += 1
        self._frames_label.setText(f"Frames: {self._frames}")

    def _on_thread_finished(self) -> None:
        self._set_running(False)
        self.thread = None

    def _set_running(self, running: bool) -> None:
        self.ip_edit.setEnabled(not running)
        self.port_edit.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if not running:
            self.status_dot.setStyleSheet(
                f"background:{STATE_COLORS['idle']}; border-radius:7px;"
            )
            self.sent_time.setText("—")
            self.sent_sub.setText("")
            self.proto_value.setText("(waiting to start)")

    # -- Shutdown -------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)

    bridge = LogBridge()
    setup_logging(bridge)
    log.info("starting %s (config: %s, logs: %s)", APP_NAME, CONFIG_FILE, LOG_FILE)

    def excepthook(exc_type, exc, tb):
        logging.getLogger().critical("uncaught exception", exc_info=(exc_type, exc, tb))
        try:
            QMessageBox.critical(None, "Unexpected error", f"{exc_type.__name__}: {exc}")
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    config = load_config()
    apply_config_to_backend(config)

    window = MainWindow(config)
    bridge.message.connect(lambda m: window.statusBar().showMessage(m))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
