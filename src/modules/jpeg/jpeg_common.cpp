/*
 * jpeg_common.cpp
 *
 *  Created on: 1.11.2013
 *      Author: neneko
 */

#include "jpeg_common.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <unordered_map>



#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

namespace std {
template<>
struct hash<J_COLOR_SPACE> {
	size_t operator()(J_COLOR_SPACE col) const {
		return std::hash<int>()(static_cast<int>(col));
	}
};
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace yuri {
namespace jpeg {


namespace {
using namespace yuri::core::raw_format;
std::unordered_map<J_COLOR_SPACE, format_t> jpeg_to_yuri_formats = {
		{JCS_GRAYSCALE, 	y8},
		{JCS_RGB, 			rgb24},
#if defined(JCS_EXTENSIONS) && defined(JCS_ALPHA_EXTENSIONS)
		{JCS_EXT_BGR,		bgr24},
		{JCS_EXT_RGBA, 		rgba32},
		{JCS_EXT_BGRA, 		bgra32},
		{JCS_EXT_ABGR, 		abgr32},
		{JCS_EXT_ARGB, 		argb32},
#endif
		{JCS_YCbCr,			yuv444},
};


}


format_t jpeg_to_yuri(J_COLOR_SPACE colspace)
{
	auto it = jpeg_to_yuri_formats.find(colspace);
	if (it == jpeg_to_yuri_formats.end()) return 0;
	return it->second;
}
J_COLOR_SPACE  yuri_to_jpeg(format_t fmt)
{
	for (const auto& f: jpeg_to_yuri_formats) {
		if (f.second == fmt) return f.first;
	}
	return JCS_UNKNOWN;
}

std::vector<format_t> get_jpeg_supported_formats()
{
	std::vector<format_t> fmts;
	for (const auto& f: jpeg_to_yuri_formats) {
		fmts.push_back(f.second);
	}
	return fmts;
}
}
}



