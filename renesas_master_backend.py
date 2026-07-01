"""renesas_master.py — stream the PC's local (IST) time to a Renesas master clock over TCP/IP.

The master clock relays the time on to the slave clocks. This client is resilient:
  * it reconnects automatically with backoff if the device is down or the link drops,
  * it times out a dead connect/send instead of hanging,
  * it enables TCP keepalive + caps retransmission so a silently-dropped peer is noticed,
  * it paces sends with a drift-free monotonic clock, aligned to the whole second.

The protocol frame ("ARN-SGPS>dd:mm:yyyy:HH:MM:SS\r", ASCII, local time) is unchanged —
it has been verified working against the device.
"""
import logging
import socket
import threading
import time
from datetime import datetime

# ── Protocol / network configuration ────────────────────────────────
DEVICE_IP   = "192.168.1.111"       # master clock static LAN IP
DEVICE_PORT = 20108                 # master clock TCP port
SOURCE_ID   = "ARN-SGPS"            # header token expected by the master
TIME_FMT    = "%d:%m:%Y:%H:%M:%S"   # dd:mm:yyyy:HH:MM:SS, local/IST — verified working
TERMINATOR  = "\r"                  # frame terminator (CR) — verified working

# ── Timing / resilience knobs ───────────────────────────────────────
SEND_INTERVAL     = 1.0   # seconds between frames (single knob for the send rate)
CONNECT_TIMEOUT   = 5.0   # seconds to wait for connect() before giving up
SEND_TIMEOUT      = 5.0   # seconds a single send may block before erroring out
RECONNECT_DELAY   = 5.0   # seconds to wait between reconnect attempts
DEAD_PEER_TIMEOUT = 30    # seconds before a silently-vanished peer is declared dead
HEARTBEAT_EVERY   = 60    # log a liveness line every N frames

log = logging.getLogger("renesas_master")


def build_payload(now: datetime) -> bytes:
    """Build one wire frame for the given timestamp.

    Pure and side-effect free, so it is trivially unit-testable, e.g.:
        build_payload(datetime(2026, 7, 1, 14, 30, 59)) == b"ARN-SGPS>01:07:2026:14:30:59\\r"
    """
    return f"{SOURCE_ID}>{now.strftime(TIME_FMT)}{TERMINATOR}".encode("ascii")


def _configure_socket(sock: socket.socket) -> None:
    """Apply timeout + keepalive so dead / half-open links are eventually noticed.

    Detection is layered because this is a write-only stream (we never recv):
      1. A peer that sends RST (e.g. the master reboots) errors the next send
         immediately -> caught by the reconnect loop.
      2. A peer that silently vanishes with no RST (cable pull, power loss) leaves
         our sends unacknowledged; TCP_MAXRT caps how long TCP retransmits before
         giving up, so the socket errors within ~DEAD_PEER_TIMEOUT seconds.
      3. Keepalive is a backstop for any idle period (e.g. if SEND_INTERVAL grows).
    """
    sock.settimeout(SEND_TIMEOUT)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "SIO_KEEPALIVE_VALS"):  # Windows: probe a silent peer sooner
        # (enable, idle_before_first_probe_ms, interval_between_probes_ms)
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10_000, 3_000))

    # Windows caps TCP retransmission time via TCP_MAXRT (optname 5, value in seconds).
    # Best-effort: silently skip if the platform does not support it.
    tcp_maxrt = getattr(socket, "TCP_MAXRT", 5)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, tcp_maxrt, DEAD_PEER_TIMEOUT)
    except OSError:
        log.debug("TCP_MAXRT not supported; relying on keepalive + send timeout")


def send_loop(sock: socket.socket, stop: threading.Event) -> None:
    """Send one frame per SEND_INTERVAL until stop is set or the socket errors.

    Sends are paced by time.monotonic() so the cadence never drifts, and the first
    send is aligned to the top of the wall-clock second so the whole-second value
    lands right on the .000 boundary. A socket error propagates to the caller, which
    reconnects.
    """
    # Align the first send to the next whole second, then hold a fixed cadence.
    stop.wait(1.0 - (time.time() % 1.0))
    next_tick = time.monotonic()
    count = 0
    while not stop.is_set():
        sock.sendall(build_payload(datetime.now()))
        count += 1
        if count % HEARTBEAT_EVERY == 0:
            log.info("streaming — %d frames sent", count)
        else:
            log.debug("sent frame %d", count)
        next_tick += SEND_INTERVAL
        stop.wait(max(0.0, next_tick - time.monotonic()))


def stream_forever(stop: threading.Event) -> None:
    """Connect, stream, and reconnect with backoff until stop is set.

    This is the entry point a desktop-app worker thread should call: create a
    threading.Event, run this on a daemon thread, and set the event to stop cleanly.
    """
    while not stop.is_set():
        try:
            with socket.create_connection(
                (DEVICE_IP, DEVICE_PORT), timeout=CONNECT_TIMEOUT
            ) as sock:
                _configure_socket(sock)
                log.info("connected to %s:%d", DEVICE_IP, DEVICE_PORT)
                send_loop(sock, stop)
        except OSError as exc:
            log.warning("connection lost (%s); reconnecting in %.0fs", exc, RECONNECT_DELAY)
            stop.wait(RECONNECT_DELAY)
    log.info("stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    stop = threading.Event()
    try:
        stream_forever(stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
