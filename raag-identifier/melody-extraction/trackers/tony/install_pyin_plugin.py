"""Install the pYIN Vamp plugin (Tony's transcription engine) on macOS.

Tony's own macOS build is a 64-bit Intel binary, and the standalone pYIN plugin
download is likewise old. The Vamp Plugin Pack 2.0 installer, however, is a
universal binary that carries a universal (x86_64 + arm64) `pyin.dylib` inside it,
stored uncompressed. Rather than run that GUI installer, this script mounts the
dmg and carves the dylib straight out of the installer binary, then drops it in
the standard per-user Vamp path -- no admin rights, nothing written into the repo.

    python trackers/tony/install_pyin_plugin.py           # download + install
    python trackers/tony/install_pyin_plugin.py --check    # just report status
"""

import argparse
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PACK_URL = ("https://github.com/vamp-plugins/vamp-plugin-pack/releases/download/"
            "v2.0/Vamp.Plugin.Pack.Installer-2.0.dmg")
INSTALLER_BIN = "Vamp Plugin Pack Installer.app/Contents/MacOS/Vamp Plugin Pack Installer"
VAMP_DIR = Path.home() / "Library" / "Audio" / "Plug-Ins" / "Vamp"
FAT_MAGIC = b"\xca\xfe\xba\xbe"


def _fat_images(blob):
    """Yield (offset, length) for every fat Mach-O image embedded in `blob`."""
    for match in re.finditer(re.escape(FAT_MAGIC), blob):
        offset = match.start()
        (n_arch,) = struct.unpack(">I", blob[offset + 4:offset + 8])
        if not 1 <= n_arch <= 8:
            continue
        end = 0
        for i in range(n_arch):
            head = offset + 8 + 20 * i
            _cpu, _sub, arch_off, arch_size, _align = struct.unpack(">iiIII", blob[head:head + 20])
            if arch_off > len(blob) or arch_size > len(blob):
                end = 0
                break
            end = max(end, arch_off + arch_size)
        if end:
            yield offset, end


def _extract_pyin(installer_path):
    """The installer embeds ~45 plugin binaries; find the one whose dylib id is pyin."""
    blob = installer_path.read_bytes()
    for offset, end in _fat_images(blob):
        image = blob[offset:offset + end]
        if len(image) > 8_000_000 or b"PYinVamp" not in image:
            continue  # skip the installer's own image and unrelated plugins
        with tempfile.NamedTemporaryFile(suffix=".dylib", delete=False) as tmp:
            tmp.write(image)
            candidate = Path(tmp.name)
        ident = subprocess.run(["otool", "-D", str(candidate)],
                               capture_output=True, text=True).stdout
        if "pyin.dylib" in ident:
            return candidate
        candidate.unlink()
    raise RuntimeError("no pyin.dylib found inside the Vamp Plugin Pack installer")


def _download(url, dest):
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def install(dmg_path=None):
    if sys.platform != "darwin":
        raise SystemExit("This installer is macOS-only; on Linux use your distro's "
                         "vamp-plugin-pack, on Windows the pack's .msi.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dmg = Path(dmg_path) if dmg_path else tmpdir / "pack.dmg"
        if not dmg_path:
            _download(PACK_URL, dmg)
        mount = tmpdir / "mnt"
        subprocess.run(["hdiutil", "attach", str(dmg), "-readonly", "-nobrowse",
                        "-mountpoint", str(mount)], check=True, capture_output=True)
        try:
            carved = _extract_pyin(mount / INSTALLER_BIN)
        finally:
            subprocess.run(["hdiutil", "detach", str(mount)], capture_output=True)

        VAMP_DIR.mkdir(parents=True, exist_ok=True)
        target = VAMP_DIR / "pyin.dylib"
        target.write_bytes(carved.read_bytes())
        target.chmod(0o755)
        carved.unlink()
        subprocess.run(["xattr", "-c", str(target)], capture_output=True)
        archs = subprocess.run(["lipo", "-archs", str(target)],
                               capture_output=True, text=True).stdout.strip()
        print(f"installed {target} ({archs})")
    check()


def check():
    try:
        import vamp
    except ImportError:
        print("vamp module NOT installed -- run: pip install --no-build-isolation vamp")
        return False
    plugins = vamp.list_plugins()
    ok = "pyin:pyin" in plugins
    print(f"vamp module ok; plugins visible: {plugins}")
    print("pyin:pyin " + ("available" if ok else "MISSING -- rerun this script without --check"))
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report install status")
    parser.add_argument("--dmg", type=str, default=None,
                        help="use an already-downloaded Vamp Plugin Pack dmg")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    install(args.dmg)
