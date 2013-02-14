/*
 * types.h
 *
 *  Created on: Oct 30, 2010
 *      Author: neneko
 */

#ifndef TYPES_H_
#define TYPES_H_
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

#ifdef _WIN32
#ifdef yuri_core_EXPORTS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
//#define EXPORT
#endif
#else
#define EXPORT
#endif

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

#if 1
using boost::shared_ptr;
using boost::shared_array;
using boost::weak_ptr;
using boost::make_shared;
using boost::mutex;
using boost::dynamic_pointer_cast;
#else
using std::shared_ptr;
#endif

}
#ifdef __GNUC__
#define YURI_UNUSED __attribute__((unused))
#else
#define YURI_UNUSED
#endif
#endif /* TYPES_H_ */
