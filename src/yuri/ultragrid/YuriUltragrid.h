/*
 * YuriUltragrid.h
 *
 *  Created on: 16.10.2013
 *      Author: neneko
 */

#ifndef YURIULTRAGRID_H_
#define YURIULTRAGRID_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/forward.h"
#include "yuri/log/Log.h"
#include "types.h"



namespace yuri {
namespace ultragrid {

codec_t yuri_to_uv(format_t);
format_t uv_to_yuri(codec_t );
std::string uv_to_string(codec_t);

core::pFrame copy_from_from_uv(const video_frame*, log::Log&);

bool copy_to_uv_frame(const core::pRawVideoFrame&, video_frame*);
}
}

// Clang issues warning about mismatch of struct/class in hash definitions somewhere in gcc stdlib...

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

inline bool operator==(const video_desc& a, const video_desc&b)
{
	return	(a.width 	== b.width) &&
			(a.height 	== b.height) &&
			(a.color_spec == b.color_spec) &&
			(a.interlacing == b.interlacing) &&
			(a.tile_count == b.tile_count);
	// Not comparing fps...

}
inline bool operator!=(const video_desc& a, const video_desc&b)
{
	return !(a==b);
}

namespace std {
template<>
struct hash<codec_t> {
 size_t operator()(const codec_t& x) const
{
	 return hash<int>()(x);
}
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}


#endif /* YURIULTRAGRID_H_ */
