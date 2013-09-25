/*
 * new_types.h
 *
 *  Created on: 30.7.2013
 *      Author: neneko
 */

#ifndef NEW_TYPES_H_
#define NEW_TYPES_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include <condition_variable>
#include <ios>
#include "platform.h"

namespace yuri {

typedef ::size_t 		size_t;
#if defined YURI_APPLE
typedef int64_t			ssize_t;
#else
typedef ::ssize_t 		ssize_t;
#endif
using	std::shared_ptr;
using	std::make_shared;
using 	std::enable_shared_from_this;
using	std::weak_ptr;
using	std::unique_ptr;
using	std::dynamic_pointer_cast;
using 	std::mutex;
typedef std::unique_lock<std::mutex> lock_t;
using	std::thread;
using 	std::function;
using 	std::condition_variable;

typedef size_t			dimension_t;
typedef ssize_t			position_t;
typedef	size_t			index_t;

struct range_t {
	ssize_t				min_value;
	ssize_t				max_value;
};

struct resolution_t {
	dimension_t	width;
	dimension_t	height;
};

struct geometry_t {
	dimension_t	width;
	dimension_t	height;
	position_t	x;
	position_t	y;
//	operator resolution_t() { return resolution_t{width, height};}
};

typedef int32_t format_t;
typedef int32_t pix_fmt_t;


enum class interlace_t {
	progressive,
	segmented_frame,
	interlaced,
	splitted
};

enum class field_order_t {
	none,
	top_field_first,
	bottom_field_first
};

inline bool operator==(const resolution_t& a, const resolution_t& b)
{
	return (a.width == b.width) && (a.height == b.height);
}
inline bool operator!=(const resolution_t& a, const resolution_t& b)
{
	return !(a==b);
}

template<class Stream>
Stream& operator<<(Stream& os, const resolution_t& res)
{
	os << res.width << "x" << res.height;
	return os;
}

template<class Stream>
Stream& operator>>(Stream& is, resolution_t& res)
{
	resolution_t r;
	char c;
	is >> r.width >> c >> r.height;
	if (c != 'x' && c != 'X') is.setstate(std::ios::failbit);
	if (!is.fail()) res = r;
	return is;
}

template<class Stream>
Stream& operator<<(Stream& os, const geometry_t& geo)
{
	os << geo.width << "x" << geo.height << "+" << geo.x << "+" << geo.y;
	return os;
}

/*!
 * Extracts geometry_t from input stream
 * It expects form of 100x200+10+20
 *
 * For negative offsets, the expected for is 100x200+-100+-20
 * @param is	Input stream
 * @param geo	Reference to geometry_t
 * @return		Input stream
 */
template<class Stream>
Stream& operator>>(Stream& is, geometry_t& geo)
{
	geometry_t g;
	char c0, c1, c2;
	is >> g.width >> c0 >> g.height >> c1 >> g.x >> c2 >> g.y;
	if ((c0 != 'x' && c0 != 'X') || c1!='+' || c2 !='+') is.setstate(std::ios::failbit);
	if (!is.fail()) geo = g;
	return is;
}

}




#endif /* NEW_TYPES_H_ */
