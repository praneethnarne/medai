"""Environment fixes that must happen before other libraries are imported.

Import this first from any entry point:

    import bootstrap  # noqa: F401

Why: a python.org build on macOS ships without a CA bundle, so anything using
urllib -- torch.hub fetching pretrained weights, the ISIC downloader -- fails
TLS verification with "unable to get local issuer certificate". Pointing
SSL_CERT_FILE at certifi's bundle fixes it for the whole process without
touching the system Python or disabling verification.
"""

import os

try:
    import certifi

    _bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", _bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _bundle)
except ImportError:  # pragma: no cover - certifi is in requirements
    pass
