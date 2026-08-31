from .gamadhani import GaMaDhaNiEmbedder
from .vinet import VINetEmbedder

REGISTRY = {
    "gamadhani": GaMaDhaNiEmbedder,
    "vinet": VINetEmbedder,
}
