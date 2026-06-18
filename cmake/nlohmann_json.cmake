#
# Add in your CMakeLists.txt
# >> include("cmake/nlohmann_json.cmake")
# >> target_link_libraries
#       ...
#       nlohmann_json::nlohmann_json
#    )
include(FetchContent)
# Prefer an already-installed nlohmann-json (e.g. brew/apt, or one pointed at via
# -Dnlohmann_json_DIR=...). This keeps offline / sandboxed cross-compiles working
# without network. If none is found, fall back to fetching the pinned release.
# Offline builds with no system package can point FetchContent at a local copy:
#   cmake -DFETCHCONTENT_SOURCE_DIR_JSON=/path/to/json ...
find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP true
    )
    FetchContent_MakeAvailable(json)
endif()
