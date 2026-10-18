#include "mmap.h"
#include <stddef.h>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>

MmapError mmap_file(FILE *const file, size_t const offset, size_t size, MappedFile *const out_mapping) {
  if (!file || !out_mapping || size == 0)
    return MMAP_ERR_INVALID_ARGUMENT;

  *out_mapping = (MappedFile){0};

  int fd = _fileno(file);
  HANDLE hFile = (HANDLE)(uintptr_t)_get_osfhandle(fd);
  if (hFile == INVALID_HANDLE_VALUE)
    return MMAP_ERR_INVALID_ARGUMENT;

  LARGE_INTEGER fileSize;
  if (!GetFileSizeEx(hFile, &fileSize))
    return MMAP_ERR_FILE_SIZE;

  if (offset >= (size_t)fileSize.QuadPart)
    return MMAP_ERR_INVALID_ARGUMENT; // offset is out of bounds

  // clamp size if it goes past the end of the file
  if (offset + size > (size_t)fileSize.QuadPart) {
    size = (size_t)fileSize.QuadPart - offset;
  }

  // get system allocation granularity for alignment
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  size_t const granularity = sysInfo.dwAllocationGranularity;

  // calculate aligned offset and adjustment
  size_t const aligned_offset = (offset / granularity) * granularity;
  size_t const adjustment = offset - aligned_offset;
  size_t const map_size = size + adjustment;

  HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!hMapping)
    return MMAP_ERR_MAPPING_FAILED;

  ULARGE_INTEGER offset_union;
  offset_union.QuadPart = aligned_offset;

  void *const ptr = MapViewOfFile(hMapping, FILE_MAP_READ, offset_union.HighPart, offset_union.LowPart, map_size);
  CloseHandle(hMapping);

  if (!ptr)
    return MMAP_ERR_MAPPING_FAILED;

  // save the base pointer for unmapping, but expose the requested user data pointer
  *out_mapping = (MappedFile){
      .map_ptr = ptr,
      .data = (void *)((uintptr_t)ptr + adjustment),
      .size = size,
      .map_size = map_size, // track size used for unmapping if needed
  };
  return MMAP_SUCCESS;
}

void munmap_file(MappedFile *const mapping) {
  if (mapping && mapping->map_ptr) {
    UnmapViewOfFile(mapping->map_ptr);
  }
}

#else
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MmapError mmap_file(FILE *const file, size_t const offset, size_t size, MappedFile *const out_mapping) {
  if (!file || !out_mapping || size == 0)
    return MMAP_ERR_INVALID_ARGUMENT;

  *out_mapping = (MappedFile){0};

  int fd = fileno(file);
  struct stat sb;
  if (fstat(fd, &sb) == -1)
    return MMAP_ERR_FILE_SIZE;

  if ((off_t)offset >= sb.st_size)
    return MMAP_ERR_INVALID_ARGUMENT;

  if ((off_t)(offset + size) > sb.st_size) {
    size = (size_t)(sb.st_size - offset);
  }

  // Get system page size for alignment
  size_t page_size = (size_t)sysconf(_SC_PAGESIZE);

  // Calculate aligned offset and adjustment
  size_t aligned_offset = (offset / page_size) * page_size;
  size_t adjustment = offset - aligned_offset;
  size_t map_size = size + adjustment;

  void *ptr = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fd, (off_t)aligned_offset);
  if (ptr == MAP_FAILED)
    return MMAP_ERR_MAPPING_FAILED;

  // Save the base pointer for unmapping, but expose the requested user data pointer
  *out_mapping = (MappedFile){
      .map_ptr = ptr,
      .data = (void *)((uintptr_t)ptr + adjustment),
      .size = size,
      .map_size = map_size, // track size used for unmapping if needed
  };
  return MMAP_SUCCESS;
}

void munmap_file(MappedFile *const mapping) {
  if (mapping && mapping->map_ptr) {
    munmap((void *)(uintptr_t)mapping->map_ptr, mapping->map_size);
  }
}
#endif
