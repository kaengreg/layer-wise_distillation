import os
import socket

import pytest


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/tmp/layerwise-distillation-tests-hf"


@pytest.fixture(autouse=True)
def block_internet(monkeypatch):
    """Fail every test that attempts an IPv4 or IPv6 connection."""
    original_connect = socket.socket.connect

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail(f"test attempted network access to {address}")
        return original_connect(sock, address)

    def guarded_create_connection(address, *args, **kwargs):
        pytest.fail(f"test attempted network access to {address}")

    def guarded_getaddrinfo(host, *args, **kwargs):
        pytest.fail(f"test attempted DNS resolution for {host}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
