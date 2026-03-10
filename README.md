# CUDA Ray Tracer

GPU-accelerated ray tracing renderer built with CUDA and SDL2

## Features

- **GPU-Accelerated Rendering**: Uses cuda for parrelism on nvidia gpus
- **Material System**: Support for multiple material types (diffuse, specular, glossy)
- **Normal Mapping**: Enhanced surface detail with normal map support
- **Texture Support**: Texture mapping with customizable UV coordinates
- **OBJ Model Loading**: Import 3D models from OBJ files (needs to be optimized)
- **BVH Acceleration**: SAH BVH for efficient ray-scene intersection testing
- **Reflections**: Multi-bounce reflection support with configurable reflection depth
- **Real-Time Interaction**: Mouse and keyboard controls for camera movement and rendering adjustments

## Project Structure

```
renderer/
├── kernel.cu             # Main entry point and scene setup
├── engine.cuh            # Full engine class
├── renderer.cuh          # Ray tracing kernel and lighting calculations
├── scene.cuh             # SOA scene class implementation
├── bvh.cuh               # BVH acceleration structure
├── objects.cuh           # Host object definitions and geometry primitives
├── light.cuh             # Light sources class
├── material.cuh          # Material properties and surface types
├── texture.cuh           # Texture management and sampling
├── normal.cuh            # Normal mapping implementation
├── algebra.cuh           # Utilities implementation mainly just math for vectors , matrix ,..
├── obj_loader.cuh        # OBJ file parsing and loading
└── settings.cuh          # Configuration and runtime settings
```

### Preview

| No Median Filter | Median Filter |
|------------------|--------------|
| ![](.\captures\no median filter.png) | ![](.\captures\median filter.png) |

## Controls

| Key | Action |
|-----|--------|
| `W/A/S/D` | Move camera forward/left/backward/right |
| `E` | Move camera up |
| `Q` | Move camera down |
| `Mouse` | Look around |
| `H` | Enable high-quality rendering (SSAA x4) |
| `L` | Toggle light movement mode |

## Configuration

Modify `settings.cuh` to adjust:
- Camera movement speed
- Mouse sensitivity
- Quality settings

Modify `kernel.cu` to:
- Load different OBJ models
- Adjust light positions and colors
- Configure material properties
- Change scene layout