"""Location of the simulator data files (catalogue, config, scenarios, schema)."""

import os
from pathlib import Path


def simulator_home() -> Path:
    """Directory holding catalogue.yaml, config/, scenarios/ and schema/.

    `SIM_HOME` overrides it (the container image sets it); the default is the `simulator/`
    directory of the source tree, two levels above this package.
    """
    override = os.environ.get("SIM_HOME")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def catalogue_path() -> Path:
    return simulator_home() / "catalogue.yaml"


def schema_path() -> Path:
    return simulator_home() / "schema" / "telemetry.schema.json"
