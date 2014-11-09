/*!
 * @file 		try_conversion.cpp
 * @author 		Zdenek Travnicek
 * @date 		9.11.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2014
 * 				CESNET z.s.p.o. 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 */

#include "try_conversion.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/raw_audio_frame_params.h"

#include "yuri_listings.h"

#include "yuri/core/thread/ConvertUtils.h"


namespace yuri {
namespace app {

namespace {
format_t parse_format(const std::string& fmt)
{
	if (auto f = core::raw_format::parse_format(fmt)) return f;
	if (auto f = core::compressed_frame::parse_format(fmt)) return f;
	if (auto f = core::raw_audio_format::parse_format(fmt)) return f;
	return 0;
}
}


void try_conversion(yuri::log::Log& l_, const std::string& format_in, const std::string& format_out)
{
	format_t f1 = parse_format(format_in);
	format_t f2 = parse_format(format_out);
	if (!f1 || !f2) {
		l_[log::error] << "Unknown format specified";
	} else try_conversion(l_, f1, f2);
}

void try_conversion(yuri::log::Log& l_,format_t format_in, format_t format_out)
{
	l_[log::info] << "Looking up path for " << get_format_name_no_throw(format_in) << " -> " << get_format_name_no_throw(format_out);
	const auto& xpath = yuri::core::find_conversion(format_in, format_out);
	if (xpath.first.empty()) {
		l_[log::error] << "No conversion found between these formats!";
	} else {
		auto ll = l_[log::info];
		ll << "Path: " << get_format_name_no_throw(format_in);
		for (const auto& x: xpath.first) {
			ll << " -> [" << x.name <<"] -> " << get_format_name_no_throw(x.target_format);
		}
	}
}


}
}

