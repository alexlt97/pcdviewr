# pcdviewr

A lightweight GPU-accelerated point cloud viewer for `.pcd` files (ASCII and binary\_compressed).

## Features

- Continuous WASD movement, Q/E down/up, Shift for faster movement
- Mouse orbit, pan, zoom, and Fly mode for looking around from a fixed position
- Open without a file and add multiple `.pcd` files with the GUI file browser
- Distinct colors for each point cloud, with matching scene-list swatches
- Per-cloud visibility and origin-frame toggles (X red, Y green, Z blue)
- Ctrl+click to select a point and show coordinates
- Adjustable movement speed and point size

Clouds use their stored XYZ coordinates without automatic alignment or transforms.
Their coordinate-origin frames therefore overlap at (0, 0, 0). PCD `VIEWPOINT`
metadata is not applied.

## Building

### Linux

```bash
# Install system dependencies (OpenGL, X11/Wayland via miniquad)
sudo apt install pkg-config libgl1-mesa-dev libglu1-mesa-dev libwayland-dev

cargo build --release
```

Binary: `target/release/pcdviewr`

---

### Windows

#### Option A — Build natively on Windows (recommended)

1. Install [Rust](https://rustup.rs) with the **MSVC** toolchain (default on Windows).
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload required by `lzf-sys`).
3. Build:

```powershell
cargo build --release
```

Binary: `target\release\pcdviewr.exe`

#### Option B — Cross-compile from Linux with MinGW

**Arch Linux:**
```bash
# Install the cross-compiler and Rust target
sudo pacman -S mingw-w64-gcc
rustup target add x86_64-pc-windows-gnu

# Build
cargo build --release --target x86_64-pc-windows-gnu
```

**Ubuntu/Debian:**
```bash
sudo apt install gcc-mingw-w64-x86-64
rustup target add x86_64-pc-windows-gnu

cargo build --release --target x86_64-pc-windows-gnu
```

Binary: `target/x86_64-pc-windows-gnu/release/pcdviewr.exe`

> **miniquad on Windows**: uses OpenGL via WGL — no extra DLLs needed on Windows 10+.

---

## Running

```bash
# Start empty, then click "Add point cloud…"
cargo run -p app

# Use another cloud instead of the bundled default
cargo run -p app -- --show-origin path/to/cloud.pcd
```

Without a path argument, pcdviewr starts with an empty scene. Pass a path to load
an initial cloud, or add clouds from the GUI after launch. The native dialog
filters for `.pcd` files. Adding clouds preserves the camera position. Use
**Fit scene** or **Home** to frame all clouds. The native picker is enabled by
default (`native-dialog` Cargo feature). Build with `--no-default-features` to
use the in-app file browser instead.

## Controls

| Action | Control |
|--------|---------|
| Forward/backward | W / S |
| Strafe left/right | A / D |
| Down/up (world Y) | Q / E |
| Move faster | Hold Shift |
| Orbit / look in Fly mode | Left drag |
| Pan | Middle drag or Shift+left drag |
| Zoom | Wheel or right drag |
| Switch Orbit/Fly | F or GUI buttons |
| Fit scene | Home |
| Select point | Ctrl+click |
| Point size | + / - or GUI slider |
| Quit | Escape |

Movement speed is adjustable in the scene panel. Keyboard movement works in
both modes. Touch supports one-finger orbit/look, two-finger pan/pinch in Orbit
mode or movement in Fly mode, and three-finger mode switching.

## CI

GitHub Actions builds and packages for **Linux** and **Windows** on every push.
Artifacts are uploaded as zip packages for each platform.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
