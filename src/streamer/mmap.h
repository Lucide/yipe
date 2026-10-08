#pragma once

#include <stddef.h>
#include <stdio.h>

typedef enum MmapError {
  MMAP_SUCCESS = 0,
  MMAP_ERR_INVALID_ARGUMENT,
  MMAP_ERR_FILE_SIZE,
  MMAP_ERR_EMPTY_FILE,
  MMAP_ERR_MAPPING_FAILED
} MmapError;

typedef struct MappedFile {
  void const *data;
  size_t size;
} MappedFile;

MmapError mmap_file(FILE *file, MappedFile *out_mapping);
void munmap_file(MappedFile *mapping);
