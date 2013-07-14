SET(YURI_SRC ${YURI_SRC} 
	core/BasicFrame.cpp
	core/BasicFrame.h
	core/BasicIOMacros.h
	core/BasicIOThread.cpp
	core/BasicIOThread.h
	core/BasicIOFilter.cpp
	core/BasicIOFilter.h
	core/BasicPipe.cpp
	core/BasicPipe.h
	core/FixedMemoryAllocator.cpp
	core/FixedMemoryAllocator.h
	core/PipeConnector.cpp
	core/PipeConnector.h
	core/pipe_types.h
	core/SocketBase.cpp
	core/SocketBase.h
	core/types.h
	core/utils.h
	
	core/ApplicationBuilder.cpp
	core/ApplicationBuilder.h
	core/Callback.cpp
	core/Callback.h
	core/config_common.h

	core/ConfigException.cpp
	core/ConfigException.h
	
	core/Instance.cpp
	core/Instance.h
	core/Parameters.cpp
	core/Parameters.h
	core/RegisteredClass.cpp
	core/RegisteredClass.h
 
	core/tinyxml/tinystr.cpp
	core/tinyxml/tinystr.h
	core/tinyxml/tinyxml.cpp
	core/tinyxml/tinyxmlerror.cpp
	core/tinyxml/tinyxml.h
	core/tinyxml/tinyxmlparser.cpp
	
	core/ThreadBase.cpp
	core/ThreadBase.h
	core/ThreadChild.cpp
	core/ThreadChild.h
	core/ThreadSpawn.cpp
	core/ThreadSpawn.h
	
	PARENT_SCOPE)
	
	#core/Config.cpp
	#core/Config.h
	