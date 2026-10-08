#include "mmap.h"

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>

MmapError mmap_file(FILE *file, MappedFile *out_mapping) {
  if (!file || !out_mapping)
    return MMAP_ERR_INVALID_ARGUMENT;

  out_mapping->data = NULL;
  out_mapping->size = 0;

  int fd = _fileno(file);
  HANDLE hFile = (HANDLE)(uintptr_t)_get_osfhandle(fd);
  if (hFile == INVALID_HANDLE_VALUE)
    return MMAP_ERR_INVALID_ARGUMENT;

  LARGE_INTEGER fileSize;
  if (!GetFileSizeEx(hFile, &fileSize))
    return MMAP_ERR_FILE_SIZE;
  if (fileSize.QuadPart == 0)
    return MMAP_ERR_EMPTY_FILE;

  HANDLE hMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!hMapping)
    return MMAP_ERR_MAPPING_FAILED;

  void *ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(hMapping);

  if (!ptr)
    return MMAP_ERR_MAPPING_FAILED;

  out_mapping->data = ptr;
  out_mapping->size = (size_t)fileSize.QuadPart;
  return MMAP_SUCCESS;
}

void munmap_file(MappedFile *mapping) {
  if (mapping && mapping->data) {
    UnmapViewOfFile(mapping->data);
    mapping->data = NULL;
    mapping->size = 0;
  }
}

#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MmapError mmap_file(FILE *file, MappedFile *out_mapping) {
  if (!file || !out_mapping)
    return MMAP_ERR_INVALID_ARGUMENT;

  out_mapping->data = NULL;
  out_mapping->size = 0;

  int fd = fileno(file);
  struct stat sb;
  if (fstat(fd, &sb) == -1)
    return MMAP_ERR_FILE_SIZE;
  if (sb.st_size == 0)
    return MMAP_ERR_EMPTY_FILE;

  void *ptr = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED)
    return MMAP_ERR_MAPPING_FAILED;

  out_mapping->data = ptr;
  out_mapping->size = (size_t)sb.st_size;
  return MMAP_SUCCESS;
}

void munmap_file(MappedFile *mapping) {
  if (mapping && mapping->data) {
    munmap((void *)mapping->data, mapping->size);
    mapping->data = NULL;
    mapping->size = 0;
  }
}
#endif
