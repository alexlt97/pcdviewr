# pcdviewr

A lightweight GPU-accelerated point cloud viewer for `.pcd` files (ASCII and binary\_compressed).

## Features

- Orbit camera (mouse drag or 1-finger touch)
- Zoom (scroll wheel or 2-finger pinch)
- Pan (2-finger drag on touch screens)
- Ctrl+click to select and print a point's coordinates
- `+`/`-` to adjust point size
- Optional origin axis frame (`--origin` flag)

---

## Building

### Linux

```bash
# Install system dependencies (OpenGL, X11/Wayland via miniquad)
sudo apt install libgl1-mesa-dev libglu1-mesa-dev

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
pcdviewr path/to/cloud.pcd

# With origin axis frame
pcdviewr --origin path/to/cloud.pcd
```

---

## Controls

### Orbit mode (default)

| Action | Mouse | Touch / Tablet |
|--------|-------|----------------|
| Orbit | Drag | 1-finger drag |
| Zoom | Scroll wheel | 2-finger pinch |
| Pan | — | 2-finger drag |
| Select point | Ctrl+click | — |
| Increase point size | `+` | — |
| Decrease point size | `-` | — |
| Switch to Fly mode | `F` | 3-finger tap |
| Quit | Q / Escape | — |

### Fly mode (`F` or 3-finger tap to enter)

Moves the camera *through* the cloud instead of orbiting around it.

| Action | Mouse/KB | Touch / Tablet |
|--------|----------|----------------|
| Look around | Drag | 1-finger drag |
| Fly forward/back | — | 2-finger drag up/down |
| Strafe left/right | — | 2-finger drag left/right |
| Switch back to Orbit | `F` | 3-finger tap |

---

## CI

GitHub Actions builds and packages for **Linux** and **Windows** on every push.
Artifacts are uploaded as zip packages for each platform.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
