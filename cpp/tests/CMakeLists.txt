include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
# Keeps googletest's runtime library choice consistent with the rest of the
# build; harmless on non-MSVC toolchains.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(cuda_db_tests
    test_request_queue.cpp
    test_scheduler.cpp
)

if(CUDA_DB_ENABLE_CUDA)
    target_sources(cuda_db_tests PRIVATE test_memory_pool.cpp)
endif()

target_link_libraries(cuda_db_tests PRIVATE
    cuda_db_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(cuda_db_tests)
