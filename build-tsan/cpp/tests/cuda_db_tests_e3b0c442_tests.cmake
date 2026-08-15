add_test([=[RequestQueueTest.DrainsAsSoonAsBatchSizeReached]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.DrainsAsSoonAsBatchSizeReached]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.DrainsAsSoonAsBatchSizeReached]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:26]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.TimesOutAndReturnsPartialBatch]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.TimesOutAndReturnsPartialBatch]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.TimesOutAndReturnsPartialBatch]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:43]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.BatchWindowStartsAtFirstArrival]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.BatchWindowStartsAtFirstArrival]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.BatchWindowStartsAtFirstArrival]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:59]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.ZeroTimeoutReturnsFirstArrivalImmediately]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.ZeroTimeoutReturnsFirstArrivalImmediately]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.ZeroTimeoutReturnsFirstArrivalImmediately]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:82]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.WaitsWhenEmptyThenReceivesPush]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.WaitsWhenEmptyThenReceivesPush]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.WaitsWhenEmptyThenReceivesPush]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:103]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.ShutdownUnblocksWaiter]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.ShutdownUnblocksWaiter]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.ShutdownUnblocksWaiter]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:120]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.SizeReflectsQueuedItemsWithoutDraining]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.SizeReflectsQueuedItemsWithoutDraining]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.SizeReflectsQueuedItemsWithoutDraining]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:139]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RequestQueueTest.ConcurrentPushesAreAllDelivered]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=RequestQueueTest.ConcurrentPushesAreAllDelivered]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RequestQueueTest.ConcurrentPushesAreAllDelivered]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_request_queue.cpp:156]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.ResolvesEachFutureWithMatchingRequestId]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.ResolvesEachFutureWithMatchingRequestId]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.ResolvesEachFutureWithMatchingRequestId]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:50]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.BatchesWithinTimeoutEvenBelowMaxBatchSize]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.BatchesWithinTimeoutEvenBelowMaxBatchSize]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.BatchesWithinTimeoutEvenBelowMaxBatchSize]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:79]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.EngineExceptionPropagatesToAllFuturesInBatch]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.EngineExceptionPropagatesToAllFuturesInBatch]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.EngineExceptionPropagatesToAllFuturesInBatch]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:102]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.PendingRequestsCompleteAcrossStop]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.PendingRequestsCompleteAcrossStop]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.PendingRequestsCompleteAcrossStop]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:125]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.StatsCountBatchesAndRequests]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.StatsCountBatchesAndRequests]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.StatsCountBatchesAndRequests]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:148]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.QueueWaitLatencyIsRecorded]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.QueueWaitLatencyIsRecorded]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.QueueWaitLatencyIsRecorded]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:186]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.LatencyStatsAreZeroBeforeFirstRequest]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.LatencyStatsAreZeroBeforeFirstRequest]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.LatencyStatsAreZeroBeforeFirstRequest]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:211]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.IdleSchedulerDoesNotSpin]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.IdleSchedulerDoesNotSpin]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.IdleSchedulerDoesNotSpin]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:228]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[SchedulerTest.MisSizedRequestFailsWithoutCorruptingBatchMates]=]  /Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests/cuda_db_tests [==[--gtest_filter=SchedulerTest.MisSizedRequestFailsWithoutCorruptingBatchMates]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SchedulerTest.MisSizedRequestFailsWithoutCorruptingBatchMates]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/cpp/tests/test_scheduler.cpp:276]==]
    WORKING_DIRECTORY [==[/Users/jeremycortez/Desktop/cpp_projects/cuda_db/build-tsan/cpp/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(cuda_db_tests_TESTS [==[RequestQueueTest.DrainsAsSoonAsBatchSizeReached]==] [==[RequestQueueTest.TimesOutAndReturnsPartialBatch]==] [==[RequestQueueTest.BatchWindowStartsAtFirstArrival]==] [==[RequestQueueTest.ZeroTimeoutReturnsFirstArrivalImmediately]==] [==[RequestQueueTest.WaitsWhenEmptyThenReceivesPush]==] [==[RequestQueueTest.ShutdownUnblocksWaiter]==] [==[RequestQueueTest.SizeReflectsQueuedItemsWithoutDraining]==] [==[RequestQueueTest.ConcurrentPushesAreAllDelivered]==] [==[SchedulerTest.ResolvesEachFutureWithMatchingRequestId]==] [==[SchedulerTest.BatchesWithinTimeoutEvenBelowMaxBatchSize]==] [==[SchedulerTest.EngineExceptionPropagatesToAllFuturesInBatch]==] [==[SchedulerTest.PendingRequestsCompleteAcrossStop]==] [==[SchedulerTest.StatsCountBatchesAndRequests]==] [==[SchedulerTest.QueueWaitLatencyIsRecorded]==] [==[SchedulerTest.LatencyStatsAreZeroBeforeFirstRequest]==] [==[SchedulerTest.IdleSchedulerDoesNotSpin]==] [==[SchedulerTest.MisSizedRequestFailsWithoutCorruptingBatchMates]==])
