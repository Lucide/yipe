# YIPE!

<div align="center">

  <img src="assets/yipe.png" alt="yipe logo"/>

  Yet another Interactive <del>Physics Engine</del> Cube Marcher (in CUDA!)

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
  -d, --particle-diameter=NUM   Diameter of the particles
  -l, --isolevel=NUM         Isolevel for generating isosurface

  -?, --help                 Give this help list
      --usage                Give a short usage message

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.
```
