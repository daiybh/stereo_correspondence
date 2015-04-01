/*!
 * @file 		string_generator.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		31. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "string_generator.h"

#ifdef HAVE_BOOST_REGEX
#include <boost/regex.hpp>
#include "yuri/core/utils/hostname.h"
#include "yuri/version.h"
#include "yuri/core/utils/frame_info.h"
#include "yuri/core/utils/global_time.h"
#include "yuri/core/frame/Frame.h"
#include "yuri/core/frame/VideoFrame.h"
#endif

namespace yuri {
namespace core {
namespace utils {

#ifndef HAVE_BOOST_REGEX
bool is_extended_generator_supported()
{
	return false;
}

std::pair<bool, bool> analyze_string_specifiers(const std::string& pattern)
{
	return std::make_pair(false, false);
}

std::string generate_string(const std::string& pattern, const yuri::core::pFrame& frame = {})
{
	return pattern;
}
#else

bool is_extended_generator_supported()
{
	return true;
}

namespace {

const boost::regex specifier_pattern ("%(0?\\d+[simM]|[simMntfFTrHDSOv%])");

template<class S1, class S2>
std::string to_s(const std::pair<S1, S2>& p)
{
	return std::string(p.first, p.second);
}

template<class Value>
std::string parse_and_replace(const std::string& spec, const Value& value)
{
	boost::regex pat ("%(0?)(\\d*)([[:alpha:]])");
	boost::smatch what;
	std::stringstream ss;
	if (boost::regex_match(spec, what, pat)) {
		const bool zero = what[1].first!=what[1].second;
		const auto count = what[2].first==what[2].second?
						0:std::stoul(to_s(what[2]));
		print_formated_value(ss, value, count, zero);
	}
	return ss.str();
}

}


std::pair<bool, bool> analyze_string_specifiers(const std::string& pattern)
{
	bool any_spec = false;
	bool seq_spec = false;
	auto beg = pattern.cbegin();
	auto end = pattern.cend();

	boost::smatch what;

	while(boost::regex_search(beg, end, what, specifier_pattern, boost::match_default)) {
		any_spec = true;

		assert (std::distance(what[0].first, what[0].second) > 0);
		auto specifier = *(what[0].second - 1);
		switch (specifier) {
			case 't':
			case 'T':
			case 's':
			case 'i':
			case 'f':
			case 'F':
			case 'm':
			case 'M':
			case 'r':
				seq_spec = true;
				break;
			default:
				break;
		}

		if (seq_spec) break;

		beg = what[0].second;
	}
	return std::make_pair(any_spec, seq_spec);
}

std::string generate_string(const std::string& pattern, index_t sequence, const yuri::core::pFrame& frame)
{
	std::string new_str;
	auto beg = pattern.cbegin();
	auto end = pattern.cend();
	boost::smatch what;
	std::stringstream ss;
	const timestamp_t current_timestamp;
	while(boost::regex_search(beg, end, what, specifier_pattern, boost::match_default)) {
		assert (std::distance(what[0].first, what[0].second) > 0);
		if (beg != what[0].first) {
			ss << std::string{beg, what[0].first};
		}
		auto specifier = *(what[0].second - 1);
		switch (specifier) {
			case 's':
				ss << parse_and_replace(to_s(what[0]), sequence);
				break;
			case 'i':
				ss << parse_and_replace(to_s(what[0]), frame->get_index());
				break;
//			case 'n':
//				ss << get_node_name();
//				break;
			case 'T':
				ss << current_timestamp;
				break;
			case 't':
				if (frame) ss << frame->get_timestamp();
				break;
			case 'm':
				if (frame) ss << parse_and_replace(to_s(what[0]),
						(frame->get_timestamp() - core::utils::get_global_start_time()).value/1000);
				break;
			case 'M':
				if (frame) ss << parse_and_replace(to_s(what[0]),
						(frame->get_timestamp() - core::utils::get_global_start_time()).value);
				break;
			case 'H':
				ss << core::utils::get_hostname();
				break;
			case 'D':
				ss << core::utils::get_domain();
				break;
			case 'S':
				ss << core::utils::get_sysver();
				break;
			case 'O':
				ss << core::utils::get_sysname();
				break;
			case 'v':
				ss << yuri_version;
				break;
			case 'f':
				if (frame) ss << core::utils::get_frame_type_name(frame->get_format(), true);
				break;
			case 'F':
				if (frame) ss << core::utils::get_frame_type_name(frame->get_format(), false);
				break;
			case 'r':
				if (auto f = std::dynamic_pointer_cast<core::VideoFrame>(frame))
					ss << f->get_resolution();
				break;
			case '%':
				ss << "%";
				break;
			default:
				break;
		}
		beg = what[0].second;
	}
	if (beg != end) {
		ss << std::string(beg, end);
	}
	return ss.str();
}
#endif


}
}
}


