#!/usr/bin/env python3
"""A llama-server for tests: the same flags, the same endpoints, no model.

It is loading for the first ``STUB_LOAD_SECONDS`` (default 0.5) seconds, during which
``/health`` answers 503 as the real one does, then it serves ``/health``,
``/v1/models``, ``/props`` and ``/v1/chat/completions`` (echoing the last message).
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]


def flag(name, default=None):
    return args[args.index(name) + 1] if name in args else default


ALIAS = flag("--alias", "stub")
MODEL = flag("-m", "stub.gguf")
PORT = int(flag("--port", "8080"))
HOST = flag("--host", "127.0.0.1")
CTX = int(flag("-c", "4096"))
READY_AT = time.monotonic() + float(os.environ.get("STUB_LOAD_SECONDS", "0.5"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if time.monotonic() < READY_AT:
            return self._send(503, {"error": {"message": "Loading model"}})
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
        if self.path == "/props":
            return self._send(200, {"model_path": MODEL, "total_slots": 4, "default_generation_settings": {"n_ctx": CTX}})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions":
            last = body.get("messages", [{}])[-1].get("content", "")
            system = next((m["content"] for m in body.get("messages", []) if m.get("role") == "system"), "")
            text = f"[{ALIAS}] {system + ' | ' if system else ''}you said: {last}"
            if body.get("stream"):
                # Server-sent events, as the real server streams: reasoning first, then words.
                pieces = [{"reasoning_content": "hmm"}] + [{"content": w} for w in ("<think>x</think>", *[t + " " for t in f"streamed {len(body['messages'])} messages; last: {last}".split(" ")])]
                finish = "stop"
                if "think forever" in last:
                    # A reasoning model that runs out of tokens before it answers.
                    pieces, finish = [{"reasoning_content": "hmm "}] * 5, "length"
                pieces = [*pieces, {}]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta in pieces:
                    chunk = {"id": "stub", "object": "chat.completion.chunk", "created": 0, "model": ALIAS, "choices": [{"index": 0, "delta": delta, "finish_reason": None if delta else finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            return self._send(
                200,
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": ALIAS,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        self._send(404, {"error": "not found"})


HTTPServer((HOST, PORT), Handler).serve_forever()
