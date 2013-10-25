/*
 * compressed_frame_params.cpp
 *
 *  Created on: 4.10.2013
 *      Author: neneko
 */

#include "compressed_frame_types.h"
#include "compressed_frame_params.h"
#include "yuri/core/utils.h"

namespace yuri {
namespace core {
namespace compressed_frame {

namespace {
	mutex	format_info_map_mutex;
	format_t last_user_format = user_start;
	comp_format_info_map_t format_info_map = {
			{jpeg, 	{jpeg,	"JPEG", {"JPG","JPEG"}, {"image/jpeg"} }},
			{mjpg, 	{mjpg,	"Motion JPEG", {"MJPG","MJPEG"}, {"video/mjpeg"} }},
			{png, 	{png,	"PNG", {"PNG"}, {"image/png"} }},
			{h264, 	{h264,	"H.264", {"H264"}, {"image/h264"} }},
			{vp8, 	{vp8,	"VP8", {"VP8"}, {"image/vp8"} }},
			{dxt1,  {dxt1,  "DXT1", {"DXT1"}, {}}},
			{dxt5,  {dxt5,  "DXT5", {"DXT5"}, {}}},

	};

	bool do_add_format(const compressed_frame_info_t& info)
	{
		auto result = format_info_map.insert({info.format, info});
		return result.second;
	}

}


bool add_format(const compressed_frame_info_t& info)
{
	lock_t _(format_info_map_mutex);
	return do_add_format(info);
}

const compressed_frame_info_t& get_format_info(format_t format)
{
	lock_t _(format_info_map_mutex);
	auto it = format_info_map.find(format);
	if (it == format_info_map.end()) throw std::runtime_error("Unknown format");
	return it->second;
}

format_t new_user_format()
{
	lock_t _(format_info_map_mutex);
	return last_user_format++;
}

format_t parse_format(const std::string& name)
{
	lock_t _(format_info_map_mutex);
	for (const auto& fmt: format_info_map) {
		for (const auto& fname: fmt.second.short_names) {
			if (iequals(fname, name)) {
				return fmt.first;
			}
		}
	}
	return unknown;
}



comp_format_info_map_t::const_iterator formats::begin() const
{
	return format_info_map.begin();
}
comp_format_info_map_t::const_iterator formats::end() const
{
	return format_info_map.end();
}

}
}
}


