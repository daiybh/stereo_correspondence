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

#ifndef YURI_USE_CXX11
#error C++11 mode is required!
#endif


#if defined _WIN32
	#define YURI_WIN 1
	#define PACK_START			__pragma(pack(push,1))
	#define PACK_END			__pragma(pack(pop))
	#define DEPRECATED
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
	#define DEPRECATED			__attribute__((deprecated))
	#ifdef __GNUG__
		#define GCC_VERSION (__GNUC__ * 10000 \
								   + __GNUC_MINOR__ * 100 \
								   + __GNUC_PATCHLEVEL__)
		#if GCC_VERSION < 40800
			// GCC 4.7.x doesn't support std::map::emplace ...
			#define EMPLACE_UNSUPPORTED  1
		#endif
	#endif
#else
	#error Unsupported platform
#endif

#ifdef YURI_ANDROID

//#include <exception>
//namespace std {
//struct bad_cast : public exception {bad_cast operator()(){}};
//struct out_of_range: public exception {out_of_range operator()(){}};
//struct runtime_error: public exception {runtime_error operator()(){}};
//}
#endif

#endif /* PLATFORM_H_ */
