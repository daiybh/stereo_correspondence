/*
 * time_types.h
 *
 *  Created on: 8.9.2013
 *      Author: neneko
 */

#ifndef TIME_TYPES_H_
#define TIME_TYPES_H_

#include <chrono>
#include <cstdlib>
#include <iomanip>
namespace yuri {
namespace detail {
	using clock_t 		= std::chrono::high_resolution_clock;
	using time_point 	= clock_t::time_point;
	using period		= std::ratio<1,1000000>;
	using duration_rep	= int64_t;
	using duration		= std::chrono::duration<duration_rep,period>;
}

struct timestamp_t {
	using duration = detail::time_point::duration;

	timestamp_t():value(detail::clock_t::now()) {}
	template<class Duration>
	timestamp_t(const std::chrono::time_point<detail::clock_t,Duration>& tp):
		value(std::chrono::time_point_cast<duration>(tp))
	{}

	detail::time_point			value;
};

inline constexpr bool operator==(const timestamp_t& a, const timestamp_t& b) {
	return a.value == b.value;
}

inline constexpr bool operator!=(const timestamp_t& a, const timestamp_t& b) {
	return a.value != b.value;
}
inline constexpr bool operator>(const timestamp_t& a, const timestamp_t& b) {
	return a.value > b.value;
}
inline constexpr bool operator>=(const timestamp_t& a, const timestamp_t& b) {
	return a.value >= b.value;
}
inline constexpr bool operator<(const timestamp_t& a, const timestamp_t& b) {
	return a.value < b.value;
}

inline constexpr bool operator<=(const timestamp_t& a, const timestamp_t& b) {
	return a.value <= b.value;
}

struct duration_t {
	constexpr duration_t():value(0) {}
	explicit constexpr duration_t(int64_t microseconds):value(microseconds){}
	template<typename T, class Period>
	explicit duration_t(const std::chrono::duration<T, Period>& dur){
		value = std::chrono::duration_cast<detail::duration>(dur).count();
	}
	template<typename Rep, class Period>
	explicit operator std::chrono::duration<Rep, Period>() const {
		return std::chrono::duration_cast<
					std::chrono::duration<Rep, Period>>(
							detail::duration(value));
	}
	detail::duration_rep		value;
};

inline constexpr duration_t operator"" _us(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val)}; }
inline constexpr duration_t operator"" _ms(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val*1e3)}; }
inline constexpr duration_t operator"" _s(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val*1e6)}; }
inline constexpr duration_t operator"" _minutes(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val*60e6)}; }
inline constexpr duration_t operator"" _hours(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val*3600e6)}; }
inline constexpr duration_t operator"" _days(unsigned long long val) { return duration_t{static_cast<detail::duration_rep>(val*24*3600e6)}; }

inline constexpr duration_t operator"" _us(long double val) { return duration_t{static_cast<detail::duration_rep>(val)}; }
inline constexpr duration_t operator"" _ms(long double val) { return duration_t{static_cast<detail::duration_rep>(val*1e3)}; }
inline constexpr duration_t operator"" _s(long double val) { return duration_t{static_cast<detail::duration_rep>(val*1e6)}; }
inline constexpr duration_t operator"" _minutes(long double val) { return duration_t{static_cast<detail::duration_rep>(val*60e6)}; }
inline constexpr duration_t operator"" _hours(long double val) { return duration_t{static_cast<detail::duration_rep>(val*3600e6)}; }
inline constexpr duration_t operator"" _days(long double val) { return duration_t{static_cast<detail::duration_rep>(val*24*3600e6)}; }

inline duration_t operator-(const timestamp_t& t1, const timestamp_t& t2)
{
	return duration_t(t1.value - t2.value);
}
inline constexpr duration_t operator-(const duration_t& a)
{
	return duration_t(-a.value);
}
inline constexpr bool operator==(const duration_t& a, const duration_t& b)
{
	return a.value == b.value;
}
inline constexpr bool operator!=(const duration_t& a, const duration_t& b)
{
	return !(a.value == b.value);
}
inline constexpr bool operator>(const duration_t& a, const duration_t& b)
{
	return a.value > b.value;
}
inline constexpr bool operator>=(const duration_t& a, const duration_t& b)
{
	return !(a.value < b.value);
}
inline constexpr bool operator<(const duration_t& a, const duration_t& b)
{
	return a.value < b.value;
}
inline constexpr bool operator<=(const duration_t& a, const duration_t& b)
{
	return !(a.value > b.value);
}
inline constexpr duration_t operator-(const duration_t& a, const duration_t& b)
{
	return duration_t(a.value-b.value);
}
inline constexpr duration_t operator+(const duration_t& a, const duration_t& b)
{
	return duration_t(a.value+b.value);
}
template<typename T>
inline constexpr
typename std::enable_if<std::is_arithmetic<T>::value,duration_t>::type
operator/(const duration_t& a, const T& b)
{
	return duration_t(a.value/b);
}
template<typename T>
inline constexpr
typename std::enable_if<std::is_arithmetic<T>::value,duration_t>::type
operator/(const T& b, const duration_t& a)
{
	return duration_t(a.value/b);
}
template<typename T>
inline constexpr
typename std::enable_if<std::is_arithmetic<T>::value,duration_t>::type
operator*(const duration_t& a, const T& b)
{
	return duration_t(a.value*b);
}
template<typename T>
inline constexpr
typename std::enable_if<std::is_arithmetic<T>::value,duration_t>::type
operator*(const T& b, const duration_t& a)
{
	return duration_t(a.value*b);
}

template<class Stream>
Stream& operator<<(Stream& os, const duration_t& duration)
{
	const uint64_t val 		= std::abs(duration.value);
	const uint64_t hours 	=  val / 1000000 / 3600;
	const uint64_t minutes 	= (val / 1000000 / 60) % 60;
	const uint64_t seconds	= (val / 1000000)% 60;
	const uint64_t useconds	=  val % 1000000;
	os << std::fixed << std::setw(2) << std::setfill('0') << hours << ":" << std::setw(2) << minutes << ":" << std::setw(2)<< seconds << "." << std::setw(6)<< useconds;
	return os;
}
/// TODO: Not implemented...
template<class Stream>
Stream& operator<<(Stream& os, const timestamp_t&)
{
	os << "Timestamp []";
	return os;
}


}
#endif /* TIME_TYPES_H_ */
