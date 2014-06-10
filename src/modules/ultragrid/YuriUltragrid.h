/*!
 * @file 		YuriUltragrid.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef YURIULTRAGRID_H_
#define YURIULTRAGRID_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/VideoFrame.h"
#include "yuri/core/forward.h"
#include "yuri/log/Log.h"
#include "uv_video.h"
#include "audio/audio.h"

namespace yuri {
namespace ultragrid {

void init_uv();

typedef std::shared_ptr<::audio_frame2> audio_frame_t;

codec_t yuri_to_uv(format_t);
format_t uv_to_yuri(codec_t );
format_t audio_uv_to_yuri(int bits_per_sample);
codec_t yuri_to_uv_compressed(format_t);
format_t uv_to_yuri_compressed(codec_t );
interlace_t uv_interlace_to_yuri(::interlacing_t);
field_order_t uv_fo_to_yuri(::interlacing_t);

std::string uv_to_string(codec_t);
std::string yuri_to_uv_string(format_t);

core::pFrame copy_from_from_uv(const video_frame *, log::Log&);
inline core::pFrame copy_from_from_uv(video_frame_t frame, log::Log& log)
{
	return copy_from_from_uv(frame.get(), log);
}

core::pFrame create_yuri_from_uv_desc(const video_desc *, size_t data_len, log::Log&);

bool copy_to_uv_frame(const core::pFrame&, video_frame*);
bool copy_to_uv_frame(const core::pRawVideoFrame&, video_frame*);
bool copy_to_uv_frame(const core::pCompressedVideoFrame&, video_frame*);

template<class T>
bool copy_to_uv_frame(const T& frame, video_frame_t uv_frame)
{
	return copy_to_uv_frame(frame, uv_frame.get());
}

//inline bool copy_to_uv_frame(const core::pFrame& frame, video_frame_t uv_frame)
//{
//	return copy_to_uv_frame(frame, uv_frame.get());
//}
//inline bool copy_to_uv_frame(const core::pRawVideoFrame& frame, video_frame_t uv_frame)
//{
//	return copy_to_uv_frame(frame, uv_frame.get());
//}
//inline bool copy_to_uv_frame2(const core::pCompressedVideoFrame& framex, video_frame_t uv_framex)
//{
//	return copy_to_uv_frame(framex, uv_framex.get());
//}


audio_frame_t allocate_uv_frame(const core::pAudioFrame&);
audio_frame_t allocate_uv_frame(const core::pRawAudioFrame&);

video_frame_t allocate_uv_frame(const core::pVideoFrame&);
video_frame_t allocate_uv_frame(const core::pRawVideoFrame&);
video_frame_t allocate_uv_frame(const core::pCompressedVideoFrame&);
}
}


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

// Clang issues warning about mismatch of struct/class in hash definitions somewhere in gcc stdlib...

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

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
