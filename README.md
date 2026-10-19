# YIPE!

<div align="center">
<img src="assets/yipe.png" alt="yipe logo"/>
<br>
Yet another Interactive <del>Physics Engine</del> Cube Marcher (in CUDA!)

<hr/>

Realtime triangulation of point cloud animations generated with [SPlisHSPlasH](https://splishsplash.physics-simulation.org/).
  
<video width="800" height="600" align="center" src="https://github.com/user-attachments/assets/9a720719-9ec6-4084-9820-a4954b217e89"></video>
<video width="800" height="600" align="center" src="https://github.com/user-attachments/assets/66e548d7-098a-4aaf-a219-33f3a4f802f5"></video>
</div>

---



```
Usage: yipe.exe [OPTION...] {random_points|test|player} [options]
yipe -- A onesweep/spatial-hashing/marching-cubes demo.

Subcommands:
  random_points   Render random points demo
  test            Run test suite
  player          Run player mode

 Common Options:
  -d, --particle-diameter=NUM   Diameter of the particles
  -i, --isolevel=NUM         Isolevel for generating isosurface

  -?, --help                 Give this help list
      --usage                Give a short usage message

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.
```
