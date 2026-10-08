include(FetchContent)

FetchContent_Declare(
    argp_standalone
    GIT_REPOSITORY https://github.com/tom42/argp-standalone.git
    GIT_TAG v1.2.0
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# off by default
# set(argp_standalone_BUILD_TESTING OFF)
# set(argp_standalone_ENABLE_WARNINGS OFF)

FetchContent_MakeAvailable(argp_standalone)

if(MSVC)
    target_compile_options(argp-standalone PRIVATE /wd4028)
endif()
