"""Connectors: adapters that turn external sources into Documents."""

from recall.connectors.base import Connector, connector_registry
from recall.connectors.filesystem import FilesystemConnector
from recall.connectors.pdf import PDFConnector

__all__ = [
    "Connector",
    "FilesystemConnector",
    "PDFConnector",
    "connector_registry",
    "create_connector",
]


def create_connector(source_type: str, **kwargs: object) -> Connector:
    """Instantiate the connector registered under ``source_type``."""
    return connector_registry.create(source_type, **kwargs)
