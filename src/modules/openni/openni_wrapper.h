/*
 * openni_wrapper.h
 *
 *  Created on: 19.2.2013
 *      Author: neneko
 */

#ifndef OPENNI_WRAPPER_H_
#define OPENNI_WRAPPER_H_
#include "yuri/io/platform.h"
#if defined YURI_LINUX
#if not defined linux
// OpenNI requires 'linux' to be defined for correct use (and it's not in gcc in some cases, in C++11, for example)
#define linux 1
#endif

// OpenNi has some poor programming techniques that trogger lots of warning on current gcc
#pragma GCC diagnostic push
#if defined __clang__
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wgnu"
#pragma clang diagnostic ignored "-Wnewline-eof"
#pragma clang diagnostic ignored "-Wextra-semi"
#endif
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#pragma GCC diagnostic ignored "-Wlong-long"
#endif
#include "OpenNI.h"
#if defined YURI_LINUX
#pragma GCC diagnostic pop
#endif


#endif /* OPENNI_WRAPPER_H_ */
