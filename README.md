# YIPE!

<div align="center">

  <img src="assets/yipe.png" alt="yipe logo"/>

  Yet another realtIme <del>Physics Engine</del> Cube Marcher (in CUDA!)

</div>

---

<div align="center">

  Realtime GPU triangulation of point cloud animations generated with [SPlisHSPlasH](https://splishsplash.physics-simulation.org/).

</div>

* *1568177 particles per frame, 300 frames ~60fps*

  <https://github.com/user-attachments/assets/b364c101-bf53-4f9a-80cf-612c44d9b4a1>
* *2423459 particles per frame, 300 frames ~60fps*

  <https://github.com/user-attachments/assets/94b03aa6-b849-4609-b937-c9cba7f422da>

---

```txt
Usage: yipe.exe [OPTION...] {random|test|player} [options]
yipe -- A onesweep/spatial-hashing/marching-cubes demo.

Subcommands:
  random          Render random points demo
  test            Run test suite
  player          Run player mode

 Common Options:
  -l, --isolevel=NUM         Isolevel for generating isosurface
  -r, --particle-radius=NUM  radius of the particles

  -?, --help                 Give this help list
      --usage                Give a short usage message

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.
```

Viewer controls (printed to `stdout`):

```txt
Mouse Middle: pan
Mouse Left: rotate
←: increase particle radius
→: decrease particle radius
↑: increase isolevel
↓: decrease isolevel
W: move camera forward
A: move camera to the left
S: move camera backward
D: move camera to the right
Space: play/pause
Period: frame-by-frame forward
```

* ```txt
  Usage: test [OPTION...]
  
    -c, --particle-count=NUM   Number of particles
    -s, --seed=NUM             Seed for random points
    -x, --room-width=NUM       Room width in metres
    -y, --room-height=NUM      Room height in metres
    -z, --room-depth=NUM       Room depth in metres
  
   Shared Options:
    -l, --isolevel=NUM         Isolevel for generating isosurface
    -r, --particle-radius=NUM  radius of the particles
  
    -?, --help                 Give this help list
        --usage                Give a short usage message
  
  Mandatory or optional arguments to long options are also mandatory or optional
  for any corresponding short options.
  ```

* ```txt
  Usage: player [OPTION...]
  
    -b, --start-frame=NUM      Index of starting frame
    -f, --framerate=NUM        Framerate in frames per second
    -F, --fragment-shader=FILE Path to fragment shader file
    -i, --input=FILE           Path to the recording file
    -k, --frameskip=NUM        Skipped frames (0 for no skipping)
    -t, --total-frames=NUM     Max number of frames (0 for all frames)
    -V, --vertex-shader=FILE   Path to vertex shader file
  
   Shared Options:
    -l, --isolevel=NUM         Isolevel for generating isosurface
    -r, --particle-radius=NUM  radius of the particles
  
    -?, --help                 Give this help list
        --usage                Give a short usage message
  
  Mandatory or optional arguments to long options are also mandatory or optional
  for any corresponding short options.
  ```

  A small recording file can be downloaded from the [release section](https://github.com/Lucide/yipe/releases/tag/latest). Recording files are `.bin` files that can be generated with the `fluid` python script located in `/pysrc` (uv python project):

  ```txt
  usage: fluid [-h] [--fps FPS] [--total-frames TOTAL_FRAMES] [--warmup WARMUP] scene_file output_bin
  
  export simulated fluid and obstacle particles
  
  positional arguments:
    scene_file            SPlisHSPlasH scene file
    output_bin            output binary file
  
  options:
    -h, --help            show this help message and exit
    --fps FPS
    --total-frames TOTAL_FRAMES
    --warmup WARMUP       simulate this many frames before recording
  ```

  An example SPlisHSPlasH run printout is available in `/assets/DoubleDamBreakWithSphere2.html`

* ```txt
  Usage: random [OPTION...]
  
    -c, --particle-count=NUM   Number of particles
    -f, --framerate=NUM        Framerate in frames per second
    -F, --fragment-shader=FILE Path to fragment shader file
    -k, --frameskip=NUM        Skipped frames (0 for no skipping)
    -s, --seed=NUM             Seed for random points
    -V, --vertex-shader=FILE   Path to vertex shader file
    -x, --room-width=NUM       Room width in metres
    -y, --room-height=NUM      Room height in metres
    -z, --room-depth=NUM       Room depth in metres
  
   Shared Options:
    -l, --isolevel=NUM         Isolevel for generating isosurface
    -r, --particle-radius=NUM  radius of the particles
  
    -?, --help                 Give this help list
        --usage                Give a short usage message
  
  Mandatory or optional arguments to long options are also mandatory or optional
  for any corresponding short options.
  ```

See the [docs](https://github.com/Lucide/yipe/blob/master/yipe.pdf) for more information.
