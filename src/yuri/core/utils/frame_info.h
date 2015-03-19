/*!
 * @file 		frame_info.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		19. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_FRAME_INFO_H_
#define SRC_YURI_CORE_UTILS_FRAME_INFO_H_

#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/raw_audio_frame_params.h"

namespace yuri {
namespace core {
namespace utils {

std::string get_frame_type_name(format_t fmt, bool short_name)
{
	try {
		auto fi = core::raw_format::get_format_info(fmt);
		if (short_name) {
			if (fi.short_names.empty()) return {};
			return fi.short_names[0];
		}
		return fi.name;
	}
	catch(...) {}
	try {
		auto fi = core::compressed_frame::get_format_info(fmt);
		if (short_name) {
			if (fi.short_names.empty()) return {};
			return fi.short_names[0];
		}
		return fi.name;
	}
	catch(...) {}
	try {
		auto fi = core::raw_format::get_format_info(fmt);
		if (short_name) {
			if (fi.short_names.empty()) return {};
			return fi.short_names[0];
		}
		return fi.name;
	}
	catch(...) {}
	return {};
}


}
}
}
#endif /* SRC_YURI_CORE_UTILS_FRAME_INFO_H_ */
