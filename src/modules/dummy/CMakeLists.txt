# Set name of the module
SET (MODULE dummy)

# Set all source files module uses
SET (SRC DummyModule.cpp
		 DummyModule.h)

# Finally set module version
SET (MODULE_VERSION 2.5.0)





# You shouldn't need to edit anything below this line 
SET (MODULE_LIB yuri2_module_${MODULE})
add_library(${MODULE_LIB} ${SRC})
target_link_libraries(${MODULE_LIB} ${LIBNAME})
SET_TARGET_PROPERTIES(${MODULE_LIB} PROPERTIES SOVERSION 2 VERSION ${MODULE_VERSION} )
install(TARGETS ${MODULE_LIB} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)

add_custom_command(TARGET ${MODULE_LIB} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE_LIB}>/$<TARGET_FILE_NAME:${MODULE_LIB}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE_LIB}>"
    )						