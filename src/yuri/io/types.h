/*
 * types.h
 *
 *  Created on: Oct 30, 2010
 *      Author: neneko
 */

#ifndef TYPES_H_
#define TYPES_H_
#include <boost/cstdint.hpp>

#ifdef __WIN32__
#ifdef BUILDDLL
#define EXPORT __declspec(dllexport)
#else
//define EXPORT __declspec(dllimport)
#define EXPORT
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

}
#ifdef __GNUC__
#define YURI_UNUSED __attribute__((unused))
#else
#define YURI_UNUSED
#endif
#endif /* TYPES_H_ */
