/*!
 * @file 		new_types.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		30.7.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
#include <ostream>
#include <istream>
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

struct coordinates_t {
	position_t	x;
	position_t	y;
};

struct resolution_t;
struct geometry_t {
	dimension_t	width;
	dimension_t	height;
	position_t	x;
	position_t	y;
	resolution_t get_resolution() const;
};

struct resolution_t {
	dimension_t	width;
	dimension_t	height;
	geometry_t get_geometry() const { return {width, height, 0, 0}; }
};

inline resolution_t geometry_t::get_resolution() const
{
	return {width, height};
}


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

constexpr inline resolution_t operator-(const resolution_t& a, const resolution_t& b)
{
	return {a.width - b.width, a.height - b.height};
}
constexpr inline resolution_t operator+(const resolution_t& a, const resolution_t& b)
{
	return {a.width + b.width, a.height + b.height};
}
constexpr inline coordinates_t operator-(const coordinates_t& a, const coordinates_t& b)
{
	return {a.x - b.x, a.y- b.y};
}
constexpr inline coordinates_t operator+(const coordinates_t& a, const coordinates_t& b)
{
	return {a.x + b.x, a.y + b.y};
}


constexpr inline coordinates_t operator+(const resolution_t& a, const coordinates_t& b)
{
	return {static_cast<position_t>(a.width) + b.x, static_cast<position_t>(a.height) + b.y};
}
constexpr inline coordinates_t operator+(const coordinates_t& b, const resolution_t& a)
{
	return {static_cast<position_t>(a.width) + b.x, static_cast<position_t>(a.height) + b.y};
}
constexpr inline coordinates_t operator-(const coordinates_t& a, const resolution_t& b)
{
	return {a.x - static_cast<position_t>(b.width), a.y - static_cast<position_t>(b.height)};
}


inline position_t geometry_max_x (const geometry_t rect) { return rect.width + rect.x;}
inline position_t geometry_max_y (const geometry_t rect) { return rect.height + rect.y;}

inline geometry_t intersection(const geometry_t& rect1, const geometry_t& rect2)
{
	geometry_t rect_out;
	rect_out.x = std::max(rect1.x, rect2.x);
	rect_out.y = std::max(rect1.y, rect2.y);
	rect_out.width = std::min(geometry_max_x(rect1),geometry_max_x(rect2))-rect_out.x;
	rect_out.height = std::min(geometry_max_y(rect1),geometry_max_y(rect2))-rect_out.y;
	return rect_out;
}
inline geometry_t intersection(const geometry_t& rect1, const resolution_t& rect2)
{
	return intersection(rect1, rect2.get_geometry());
}
inline geometry_t intersection(const resolution_t& rect1, const geometry_t& rect2)
{
	return intersection(rect1.get_geometry(), rect2);
}
inline resolution_t intersection(const resolution_t& rect1, const resolution_t& rect2)
{
	return {std::min(rect1.width, rect2.width), std::min(rect1.height, rect2.height)};
}

//template<class Stream>
inline std::ostream& operator<<(std::ostream& os, const resolution_t& res)
{
	os << res.width << "x" << res.height;
	return os;
}

//template<class Stream>
inline std::istream& operator>>(std::istream& is, resolution_t& res)
{
	resolution_t r;
	char c;
	is >> r.width >> c >> r.height;
	if (c != 'x' && c != 'X') is.setstate(std::ios::failbit);
	if (!is.fail()) res = r;
	return is;
}
//template<class Stream>
inline std::ostream& operator<<(std::ostream& os, const coordinates_t& res)
{
	os << res.x<< "x" << res.y ;
	return os;
}

//template<class Stream>
inline std::istream& operator>>(std::istream& is, coordinates_t& res)
{
	coordinates_t r;
	char c;
	is >> r.x >> c >> r.y;
	if (c != 'x' && c != 'X' && c!=',') is.setstate(std::ios::failbit);
	if (!is.fail()) res = r;
	return is;
}
//template<class Stream>
inline std::ostream& operator<<(std::ostream& os, const geometry_t& geo)
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
//template<class Stream>
inline std::istream& operator>>(std::istream& is, geometry_t& geo)
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
