add_subdirectory(yuri)
add_subdirectory(apps)
add_subdirectory(modules)

if (NOT YURI_DISABLE_TESTS)
	add_subdirectory(tests)
endif()	