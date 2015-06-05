/*!
 * @file 		flycap_defs.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		5. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_FLYCAP_FLYCAP_DEFS_H_
#define SRC_MODULES_FLYCAP_FLYCAP_DEFS_H_
#include "flycap_c_helpers.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri {
namespace flycap {

namespace {
using namespace core::raw_format;

struct cmp_resolution {
bool operator()(const resolution_t& a, const resolution_t& b) const
{
	return (a.width < b.width) || ((a.width==b.width) && (a.height < b. height));
}
};
const std::map<resolution_t, std::map<format_t, fc2VideoMode>, cmp_resolution> video_modes = {
		{{160,120}, {
				{yuv444, FC2_VIDEOMODE_160x120YUV444}},
		},
		{{320,240}, {
				{yuyv422, FC2_VIDEOMODE_320x240YUV422}},
		},
		{{640, 480}, {
				{y8, FC2_VIDEOMODE_640x480Y8},
				{y16, FC2_VIDEOMODE_640x480Y16},
				{rgb24, FC2_VIDEOMODE_640x480RGB},
				{yuv411, FC2_VIDEOMODE_640x480YUV411},
				{yuyv422, FC2_VIDEOMODE_640x480YUV422}},
		},
		{{800, 600}, {
				{y8, FC2_VIDEOMODE_800x600Y8},
				{y16, FC2_VIDEOMODE_800x600Y16},
				{rgb24, FC2_VIDEOMODE_800x600RGB},
				{yuyv422, FC2_VIDEOMODE_800x600YUV422}},
		},
		{{1024, 768}, {
				{y8, FC2_VIDEOMODE_1024x768Y8},
				{y16, FC2_VIDEOMODE_1024x768Y16},
				{rgb24, FC2_VIDEOMODE_1024x768RGB},
				{yuyv422, FC2_VIDEOMODE_1024x768YUV422}},
		},
		{{1280, 960}, {
				{y8, FC2_VIDEOMODE_1280x960Y8},
				{y16, FC2_VIDEOMODE_1280x960Y16},
				{rgb24, FC2_VIDEOMODE_1280x960RGB},
				{yuyv422, FC2_VIDEOMODE_1280x960YUV422}},
		},
		{{1600, 1200}, {
				{y8, FC2_VIDEOMODE_1600x1200Y8},
				{y16, FC2_VIDEOMODE_1600x1200Y16},
				{rgb24, FC2_VIDEOMODE_1600x1200RGB},
				{yuyv422, FC2_VIDEOMODE_1600x1200YUV422}},
		},

};

const std::map<fc2PixelFormat, format_t> flycap_formats = {
		{FC2_PIXEL_FORMAT_MONO8, core::raw_format::y8},
		{FC2_PIXEL_FORMAT_MONO16, core::raw_format::y16},
		{FC2_PIXEL_FORMAT_RGB8, core::raw_format::rgb24},
		{FC2_PIXEL_FORMAT_RGB16, core::raw_format::rgb48},
		{FC2_PIXEL_FORMAT_422YUV8, core::raw_format::uyvy422},
		{FC2_PIXEL_FORMAT_444YUV8, core::raw_format::yuv444},
		{FC2_PIXEL_FORMAT_RAW8, core::raw_format::bayer_rggb}
};


const std::map<size_t, fc2FrameRate> frame_rates = {
		{15, FC2_FRAMERATE_15},
		{30, FC2_FRAMERATE_30},
		{60, FC2_FRAMERATE_60},
		{120, FC2_FRAMERATE_120},
		{240, FC2_FRAMERATE_240},
};

inline fc2VideoMode get_mode(resolution_t res, format_t fmt)
{
	auto it = video_modes.find(res);
	if (it == video_modes.end()) return FC2_NUM_VIDEOMODES;
	auto it2 = it->second.find(fmt);
	if (it2 == it->second.end()) return FC2_NUM_VIDEOMODES;
	return it2->second;
}

inline fc2FrameRate get_fps(size_t fps)
{
	auto it = frame_rates.find(fps);
	if (it == frame_rates.end()) return FC2_NUM_FRAMERATES;
	return it->second;
}

inline format_t get_yuri_format(const fc2PixelFormat& fmt)
{
	auto it = flycap_formats.find(fmt);
	if (it == flycap_formats.end()) return core::raw_format::unknown;
	return it->second;
}

inline fc2PixelFormat get_fc_format(const format_t& fmt)
{
	for (auto it: flycap_formats) {
		if (it.second == fmt) {
			return it.first;
		}
	}
	return FC2_UNSPECIFIED_PIXEL_FORMAT;
}

}


}
}



#endif /* SRC_MODULES_FLYCAP_FLYCAP_DEFS_H_ */
