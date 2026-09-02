#include "hashtable/hashtable.cuh"
#include "scan/scan.cuh"
// #include "sort/sort.cuh"

GPU HOST int3 point_discretize(Point const &point) {
  return {static_cast<int>(floorf(point.x / PARTICLE_DIAMETER)), static_cast<int>(floorf(point.y / PARTICLE_DIAMETER)),
          static_cast<int>(floorf(point.z / PARTICLE_DIAMETER))};
}

GPU HOST u32 point_hash(Point const *const point, u32 const particle_count) {
  static u32 const primes[3] = {73856093, 19349663, 83492791};
  int3 discrete = point_discretize(*point);
  return ((static_cast<u32>(discrete.x) * primes[0]) ^ (static_cast<u32>(discrete.y) * primes[1]) ^
          (static_cast<u32>(discrete.z) * primes[2])) %
         particle_count;
}

extern "C" void launch_exclusive_scan(GPUInfo const *const info, u32 *const data, u32 const length) {
  scan(info, data, length);
}

extern "C" void launch_sort_points(GPUInfo const *const info, Point *const points, u32 const length) {

  auto hasher = [=] GPU(Point const &p) -> u32 { return point_hash(&p, length); };

  sort(info, points, length, hasher);
}
