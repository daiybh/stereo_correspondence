/*!
 * @file 		platform.h
 * @author 		Zdenek Travnicek
 * @date 		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 * @brief		Platform detection and platform specific defines
 */

#ifndef PLATFORM_H_
#define PLATFORM_H_


#if defined _WIN32
	#define YURI_WIN 1
	#define PACK_START			__pragma(pack(push,1))
	#define PACK_END			__pragma(pack(pop))
	#ifdef yuri_core_EXPORTS
		#define EXPORT			__declspec(dllexport)
		#define IMPORT			__declspec(dllimport)
	#else
		#define EXPORT 			__declspec(dllimport)
		#define IMPORT 			__declspec(dllexport)
	#endif
	// Disable bad macros from windows.h
	#define WIN32_MEAN_AND_LEAN
	#define NOMINMAX
#elif defined __linux__
	#define YURI_LINUX 1
	#define EXPORT
	#define IMPORT
	#define PACK_START
	#define PACK_END			__attribute__((packed))
#else
	#error Unsupported platform
#endif


#endif /* PLATFORM_H_ */
