"""Code shared across the raag-identifier subprojects.

Anything imported by more than one subproject lives here rather than inside whichever one
wrote it first. Importing this package pulls in nothing heavy -- no torch, no librosa, no
network -- so `from utils import config` is cheap enough for any entry point.

    utils.config        where external things are, one named entry each, env-overridable
    utils.dataset       fetch a pinned Hub revision to disk; read it back as Clips
    utils.raagdb        the libmogra raag database: scales, phrases, swar vocabulary
    utils.raagspace     raag-to-raag affinity, built from the database
    utils.musical_eval  graded metrics -- a wrong answer next door scores above a wrong
                        answer across the map
    utils.extract       batch pitch-tracking to a cache (needs melody-extraction)

`utils.extract` is the only module with a heavy import chain, so it is not re-exported
here; import it directly when you need it.
"""

from . import config, dataset, musical_eval, raagdb, raagspace  # noqa: F401

__all__ = ["config", "dataset", "musical_eval", "raagdb", "raagspace"]
