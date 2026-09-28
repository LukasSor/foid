"""Shared room-control catalog and JSON-over-TCP protocol for Foid."""

from foid_protocol.catalog import Catalog, Device, DeviceAction, default_devices_dir, load_catalog
from foid_protocol.codec import decode_line, encode
from foid_protocol.messages import Ack, Command, Hello, Snapshot, StateEvent, parse_message
from foid_protocol.tcp import LineClient, LineServer, DEFAULT_PORT

__all__ = [
    "Ack",
    "Catalog",
    "Command",
    "DEFAULT_PORT",
    "Device",
    "DeviceAction",
    "Hello",
    "LineClient",
    "LineServer",
    "Snapshot",
    "StateEvent",
    "decode_line",
    "default_devices_dir",
    "encode",
    "load_catalog",
    "parse_message",
]
