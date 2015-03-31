/*!
 * @file 		string_generator.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		31. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_STRING_GENERATOR_H_
#define SRC_YURI_CORE_UTILS_STRING_GENERATOR_H_
#include "yuri/core/forward.h"
#include <iomanip>
namespace yuri {
namespace core {
namespace utils {


/*
 * If compiled with boost regex support, then following format specifiers
 * are supported inside of the string:
 *
 * 	%n - module name (currently unsupported)
 * 	%t - timestamp from the incomming frame
 * 	%f - frame type (short)
 * 	%F - frame type (long)
 * 	%T - timestamp at the time of dump
 * 	%r - resolution of video frame
 * 	%m - milliseconds since start
 * 	%M - microseconds since start
 * 	%H - local hostname
 * 	%D - local domain
 * 	%s - sequence number
 * 	%i - frame index
 * 	%S - system name (eg. Linux-3.19.0)
 * 	%O - OS (eg. Linux)
 * 	%v - yuri version
 * 	%% - literal %
 *
 */


/* **************************************************
 *  NOTE: The API for string generator is by no means stable
 *  and it can change anytime!
 ************************************************** */

/*!
 *
 * @return true if extended string generator is supported
 */
bool is_extended_generator_supported();

std::pair<bool, bool> analyze_string_specifiers(const std::string& pattern);

std::string generate_string(const std::string& pattern, index_t sequence = 0, const yuri::core::pFrame& frame = {});


template<class Stream, class Value>
Stream& print_formated_value(Stream& os, const Value& value, int width = 0, bool zero = true)
{
	if (zero) {
		os << std::setfill('0');
	}
	if (width > 0) {
		os << std::setw(width);
	}
	os << value;
	return os;
}



}
}
}



#endif /* SRC_YURI_CORE_UTILS_STRING_GENERATOR_H_ */
