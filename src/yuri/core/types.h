/*!
 * @file 		BasicIOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		30.10.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef TYPES_H_
#define TYPES_H_
#include "yuri/core/platform.h"
#include "yuri/core/uvector.h"
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

//! All yuri related stuff belongs here
namespace yuri {
typedef boost::uint64_t usize_t;
typedef usize_t size_t;
typedef boost::int64_t ssize_t;
typedef boost::uint32_t uint_t;
typedef boost::int32_t sint_t;
typedef boost::uint16_t ushort_t;
typedef boost::int16_t sshort_t;
typedef boost::uint8_t ubyte_t;
typedef boost::int8_t sbyte_t;

typedef ssize_t format_t;
typedef size_t threadId_t;

typedef yuri::uvector<yuri::ubyte_t,false> plane_t;

#if 1
using boost::shared_ptr;
using boost::weak_ptr;
using boost::make_shared;
using boost::mutex;
using boost::dynamic_pointer_cast;
using boost::thread;
#else
using std::shared_ptr;
#endif


}
#ifdef YURI_LINUX
#define YURI_UNUSED __attribute__((unused))
#else
#define YURI_UNUSED
#endif
#endif /* TYPES_H_ */
