import argparse
import sys
import time
from collections.abc import Iterable
from io import BufferedWriter
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pysplishsplash as sph
from numpy.typing import NDArray
from rich.console import Console
from rich.progress import (
  BarColumn,
  Progress,
  TextColumn,
  TimeElapsedColumn,
)
from rich.table import Column

__all__ = ["main"]

# pysplishsplash pain in the ass

console = Console(highlight=False)

total_progress = Progress(
  TimeElapsedColumn(table_column=Column(justify="right", min_width=7)),
  BarColumn(
    complete_style="blue",
  ),
  TextColumn("{task.description}"),
  console=console,
)
total_task = total_progress.add_task("description", start=False, visible=False)


def infer_room_size(scene):
  room_config = next(config for config in scene.boundaryModels if config.isWall)
  return np.asarray(room_config.scale, dtype=np.float32)


def iter_non_wall_boundary_configs(scene):
  return (item for item in enumerate(scene.boundaryModels) if not item[1].isWall)


def load_boundary_cloud(scene_dir: Path, config) -> NDArray[np.float32]:
  if not config.samplesFile:
    raise ValueError(f"non-wall boundary {config.id} has no particleFile")

  particle_file: Final = (scene_dir / config.samplesFile).resolve()
  positions: Iterable[tuple[float, float, float]] = sph.Utilities.VecVector3r()
  loaded: Final = sph.Utilities.PartioReaderWriter.readParticle(
    str(particle_file),
    np.zeros(3),
    np.eye(3),
    1.0,
    positions,
  )
  if not loaded:
    raise RuntimeError(f"could not load boundary particles: {particle_file}")

  points: Final = np.asarray([[point[0], point[1], point[2]] for point in positions], dtype=np.float32).reshape((-1, 3))
  scale: Final = np.asarray(config.scale, dtype=np.float32)
  return np.ascontiguousarray(points * scale)


def export_fluid_and_obstacles(
  scene_path: Path,
  output_stream: BufferedWriter,
  frame_dt: float = 1.0 / 60,
  total_frames: int = 5 * 60,
):
  base: Final = sph.Exec.SimulatorBase()

  try:
    # initialize SPlisHSPlasH simulator
    base.init(sceneFile=str(scene_path), useGui=False)
    base.initSimulation()
    scene: Final = sph.Exec.SceneConfiguration.getCurrent().getScene()
    sim: Final = sph.Simulation.getCurrent()
    time_manager: Final = sph.TimeManager.getCurrent()
    fluid_models: Final = [sim.getFluidModel(index) for index in range(sim.numberOfFluidModels())]

    base.setValueFloat(base.STOP_AT, total_frames * frame_dt)

    fc = 1
    total_particles = 0
    boundary_bindings: list[tuple[NDArray[np.float32], Any]] = []

    # fetch lazily initialized data
    def bind_boundaries():
      nonlocal boundary_bindings
      boundary_clouds = [
        (index, load_boundary_cloud(scene_path.parent, config)) for index, config in iter_non_wall_boundary_configs(scene)
      ]
      boundary_count = sim.numberOfBoundaryModels()
      for index, points in boundary_clouds:
        if not 0 <= index < boundary_count:
          raise RuntimeError("a non-wall scene boundary has no initialized SPH boundary model")
        boundary_bindings.append(
          (
            points,
            sim.getBoundaryModel(index).getRigidBodyObject(),
          )
        )

      for index, points in boundary_clouds:
        console.print(f"boundary {index}: {len(points)} particles")

      base.setTimeStepCB(dump_frames)
      dump_frames()

    def dump_frames():
      nonlocal fc, total_particles
      # one timestep can span multiple frames
      # capture starts at frame_dt, there is no t=0 frame.
      while fc <= total_frames and fc * frame_dt <= time_manager.getTime():
        particle_blocks = []

        for fluid_model in fluid_models:
          particle_blocks.append(
            np.asarray(
              fluid_model.getFieldBuffer("position"),
              dtype=np.float32,
            )[: fluid_model.numActiveParticles()]
          )

        for points, body in boundary_bindings:
          rotation = np.asarray(body.getWorldSpaceRotation(), dtype=np.float32)
          translation = np.asarray(body.getWorldSpacePosition(), dtype=np.float32).reshape(3)
          particle_blocks.append(points @ rotation.T + translation)

        merged_frame = np.vstack(particle_blocks).astype(np.float32)

        if fc == 1:
          total_particles = merged_frame.shape[0]
          room_dimensions = infer_room_size(scene)
          output_stream.write(room_dimensions)
          output_stream.write(
            np.asarray(
              [total_frames, total_particles],
              dtype=np.dtype("=u4"),
            )
          )
          console.print("room size: " + "x".join(str(dimension) for dimension in room_dimensions))
          console.print(f"particle count: {total_particles}")

        assert fc == 1 or merged_frame.shape[0] == total_particles, (
          "particle count changed; raw fixed-size frame output is invalid"
        )
        output_stream.write(merged_frame.tobytes())
        total_progress.update(
          total_task, visible=True, completed=fc, total=total_frames, description=f"frame {fc}/{total_frames}"
        )
        fc += 1

    base.setTimeStepCB(bind_boundaries)
    with total_progress:
      base.runSimulation()

    assert fc == total_frames + 1, "simulation ended before all frames were captured"
  finally:
    base.cleanup()


def main():
  try:
    parser = argparse.ArgumentParser(description="export simulated fluid and obstacle particles")
    parser.add_argument("scene_file", help="SPlisHSPlasH scene file")
    parser.add_argument("output_bin", help="output binary file")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--total-frames", type=int, default=300)
    args = parser.parse_args()

    scene_path = Path(args.scene_file).absolute()
    if not scene_path.is_file():
      raise FileNotFoundError(f"scene file: {scene_path}")

    total_progress.start_task(total_task)
    start_time: Final = time.perf_counter()

    with open(args.output_bin, "wb") as output_stream:
      export_fluid_and_obstacles(
        scene_path,
        output_stream,
        frame_dt=1.0 / args.fps,
        total_frames=args.total_frames,
      )

    print(f"export complete in {time.perf_counter() - start_time:.2f}s.")
  except KeyboardInterrupt:
    console.print("Control-C", style="logging.level.error")
    sys.exit(0)
