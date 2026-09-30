include(FetchContent)

FetchContent_Declare(
    argp_standalone
    GIT_REPOSITORY https://github.com/tom42/argp-standalone.git
    GIT_TAG v1.2.0
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

set(argp_standalone_BUILD_TESTING OFF)

FetchContent_MakeAvailable(argp_standalone)
