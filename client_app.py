from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod

from config import ExpConfig
from federation import client_fn

if ExpConfig.FL_AGGREGATION_TYPE == "secure":
    mods = [
        secaggplus_mod,
    ]
else:
    mods = None

# Flower ClientApp
app = ClientApp(client_fn=client_fn, mods=mods)
