"""kaller, driven by a local llama.cpp model.

kaller (``github.com/jhnnsrs/kaller``) is the assistant: it finds installed actions,
calls them as held children the user approves, reads the backends, remembers the
user and asks questions. Everything it does goes through the clients it is handed,
the model included. This package hands it a local one:

- ``settings`` and ``server`` start and watch ``llama-server`` for the configured GGUF;
- ``client`` is the OpenAI client kaller talks to, pointed at that server;
- ``turn`` runs one of kaller's turns for a room message.
"""
