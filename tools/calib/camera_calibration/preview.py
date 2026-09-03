"""Dependency-free browser preview for an RTSP camera."""

from __future__ import annotations

import html
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

from .capture import CameraConfig, CaptureError, RtspCapture


class _FrameBuffer:
    """Store only the newest JPEG so slow clients cannot add latency."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def latest(self) -> tuple[int, bytes | None]:
        with self._condition:
            return self._sequence, self._jpeg

    def wait_for_next(
        self,
        sequence: int,
        timeout: float = 5.0,
    ) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != sequence,
                timeout=timeout,
            )
            return self._sequence, self._jpeg


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        frame_buffer: _FrameBuffer,
        camera_address: str,
    ) -> None:
        self.frame_buffer = frame_buffer
        self.camera_address = camera_address
        super().__init__(server_address, _PreviewRequestHandler)


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    server: _PreviewHttpServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.partition("?")[0]
        if path in {"/", "/index.html"}:
            self._send_index()
        elif path == "/stream.mjpg":
            self._send_stream()
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        elif path == "/healthz":
            self._send_health()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send_index(self) -> None:
        camera_address = html.escape(self.server.camera_address)
        page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>相机实时预览</title>
  <style>
    html, body {{ height: 100%; }}
    body {{
      box-sizing: border-box; margin: 0; padding: 20px; color: #eef2f7;
      background: #111820; font: 15px system-ui, sans-serif;
      display: flex; flex-direction: column; gap: 12px;
    }}
    header {{ display: flex; justify-content: space-between; gap: 16px; }}
    .camera {{ color: #9caabd; }}
    main {{ min-height: 0; flex: 1; display: grid; place-items: center; }}
    img {{
      display: block; max-width: 100%; max-height: 100%; object-fit: contain;
      background: #050709; border-radius: 8px;
    }}
  </style>
</head>
<body>
  <header><span>相机实时预览</span><span class="camera">{camera_address}</span></header>
  <main><img src="/stream.mjpg" alt="实时相机画面"></main>
</body>
</html>
"""
        payload = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        sequence = -1
        try:
            while True:
                sequence, jpeg = self.server.frame_buffer.wait_for_next(sequence)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except OSError:
            return

    def _send_snapshot(self) -> None:
        _sequence, jpeg = self.server.frame_buffer.latest()
        if jpeg is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No camera frame yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def _send_health(self) -> None:
        _sequence, jpeg = self.server.frame_buffer.latest()
        status = HTTPStatus.OK if jpeg is not None else HTTPStatus.SERVICE_UNAVAILABLE
        payload = b"ok\n" if jpeg is not None else b"waiting for camera\n"
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        """Keep per-frame browser requests out of terminal output."""


def encode_preview_frame(
    frame: np.ndarray,
    *,
    quality: int = 80,
    max_width: int = 1280,
) -> bytes:
    """Resize and JPEG-encode a frame for browser delivery."""
    if not 1 <= quality <= 100:
        raise ValueError("Preview JPEG quality must be between 1 and 100")
    if max_width <= 0:
        raise ValueError("Preview maximum width must be positive")
    if frame is None or not frame.size or frame.ndim < 2:
        raise CaptureError("Cannot preview an empty camera frame")

    height, width = frame.shape[:2]
    if width > max_width:
        resized_height = max(1, round(height * max_width / width))
        frame = cv2.resize(
            frame,
            (max_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise CaptureError("Failed to encode a camera frame for browser preview")
    return encoded.tobytes()


def browser_preview(
    config: CameraConfig,
    *,
    bind_host: str = "127.0.0.1",
    port: int = 8765,
    fps: float = 10.0,
    quality: int = 80,
    max_width: int = 1280,
) -> None:
    """Serve the camera as a low-latency MJPEG stream until interrupted."""
    if not bind_host.strip():
        raise ValueError("Preview bind address must not be empty")
    if not 1 <= port <= 65535:
        raise ValueError("Preview port must be between 1 and 65535")
    if fps <= 0:
        raise ValueError("Preview frame rate must be positive")

    frame_buffer = _FrameBuffer()
    try:
        server = _PreviewHttpServer(
            (bind_host, port),
            frame_buffer,
            config.address(),
        )
    except OSError as exc:
        raise CaptureError(
            f"Cannot start preview server on {bind_host}:{port}: {exc}"
        ) from exc

    http_thread = threading.Thread(
        target=server.serve_forever,
        name="camera-preview-http",
        daemon=True,
    )
    http_started = False
    try:
        with RtspCapture(config) as stream:
            frame = stream.read()
            frame_buffer.publish(
                encode_preview_frame(
                    frame,
                    quality=quality,
                    max_width=max_width,
                )
            )
            http_thread.start()
            http_started = True
            display_host = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host
            print(f"Browser preview: http://{display_host}:{server.server_port}/")
            if bind_host in {"0.0.0.0", "::"}:
                print("Preview is listening on all network interfaces; stop with Ctrl+C.")
            else:
                print("Preview is local to this machine; stop with Ctrl+C.")

            interval = 1.0 / fps
            next_publish = time.monotonic() + interval
            while True:
                frame = stream.read()
                now = time.monotonic()
                if now < next_publish:
                    continue
                frame_buffer.publish(
                    encode_preview_frame(
                        frame,
                        quality=quality,
                        max_width=max_width,
                    )
                )
                next_publish = max(next_publish + interval, now + interval)
    except KeyboardInterrupt:
        print("\nBrowser preview stopped.")
    finally:
        if http_started and http_thread.is_alive():
            server.shutdown()
        server.server_close()
        if http_thread.is_alive():
            http_thread.join(timeout=2.0)
