/*!
 * @file 		trace_method.h
 * @author 		Zdenek Travnicek
 * @date 		14.4.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef TRACE_METHOD_H_
#define TRACE_METHOD_H_


#include "platform.h"

#ifdef YURI_WIN
	#define TRACE_METHOD
#else
	#ifdef NDEBUG
		#define TRACE_METHOD
	#else
		#define TRACE_METHOD log[log::trace] << __PRETTY_FUNCTION__ << " @ " << __FILE__ << ":" << __LINE__;
	#endif
#endif


#endif /* TRACE_METHOD_H_ */
