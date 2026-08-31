#include <cuda/std/bit>
#include "sort/sort.cuh"

gpu host int3 point_discretize(Point const &point) {
  return {static_cast<int>(floorf(point.x / PARTICLE_DIAMETER)), static_cast<int>(floorf(point.y / PARTICLE_DIAMETER)),
          static_cast<int>(floorf(point.z / PARTICLE_DIAMETER))};
}

gpu host u32 point_hash(Point const *const point, u32 const particle_count) {
  static u32 const primes[3] = {73856093, 19349663, 83492791};
  int3 discrete = point_discretize(*point);
  return ((static_cast<u32>(discrete.x) * primes[0]) ^ (static_cast<u32>(discrete.y) * primes[1]) ^
          (static_cast<u32>(discrete.z) * primes[2])) %
         particle_count;
}

extern "C" void launch_sort_points(GPUInfo const *const info, Point *const points, u32 const length) {

  auto hasher = [=] gpu(Point const &p) -> u32 { return point_hash(&p, length); };

  sort(info, points, length, hasher);
}
