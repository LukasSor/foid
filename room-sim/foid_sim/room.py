from __future__ import annotations

from copy import deepcopy
from typing import Any

from foid_protocol.catalog import Catalog


class Room:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.state: dict[str, dict[str, Any]] = catalog.initial_state()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.state)

    def apply(self, device_id: str, action_id: str) -> dict[str, Any]:
        return self.catalog.apply(self.state, device_id, action_id)

    def public(self) -> dict[str, Any]:
        devices = []
        for device_id in self.catalog.order:
            device = self.catalog.get(device_id)
            devices.append(
                {
                    "id": device.id,
                    "name": device.name,
                    "ui": device.ui,
                    "state": dict(self.state.get(device_id, {})),
                    "actions": [{"id": action.id, "label": action.aliases[0]} for action in device.actions],
                }
            )
        return {"devices": devices}
