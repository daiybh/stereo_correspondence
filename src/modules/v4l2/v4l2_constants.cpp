/*!
 * @file 		v4l2_constants.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.1.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */


namespace {
using namespace yuri::core::raw_format;
using namespace yuri::core::compressed_frame;
std::map<yuri::format_t, uint32_t> formats_map= {

		{rgb24,		V4L2_PIX_FMT_RGB24},
		{argb32, 	V4L2_PIX_FMT_RGB32},
		{bgr24, 	V4L2_PIX_FMT_BGR24},
		{bgra32, 	V4L2_PIX_FMT_BGR32},
		{rgb15,		V4L2_PIX_FMT_RGB555},
		{rgb16,		V4L2_PIX_FMT_RGB565},
		{yuyv422, 	V4L2_PIX_FMT_YUYV},
		{yvyu422, 	V4L2_PIX_FMT_YVYU},
		{uyvy422, 	V4L2_PIX_FMT_UYVY},
		{vyuy422, 	V4L2_PIX_FMT_VYUY},
		//{yuv420p, V4L2_PIX_FMT_YUV420},
//			{YURI_VIDEO_DV, V4L2_PIX_FMT_DV},
		{bayer_bggr,V4L2_PIX_FMT_SBGGR8},
		{bayer_rggb,V4L2_PIX_FMT_SRGGB8},
		{bayer_grbg,V4L2_PIX_FMT_SGRBG8},
		{bayer_gbrg,V4L2_PIX_FMT_SGBRG8},

		{mjpg, 		V4L2_PIX_FMT_MJPEG},
		{jpeg, 		V4L2_PIX_FMT_JPEG},
		{h264, 		V4L2_PIX_FMT_H264}};


std::map<std::string, uint32_t> special_formats = {
	{"S920", V4L2_PIX_FMT_SN9C20X_I420},
	{"BA81", V4L2_PIX_FMT_SBGGR8}};

/** Converts yuri::format_t to v4l2 format
 * \param fmt V4l2 pixel format
 * \return yuri::format_t for the specified format.
 */
uint32_t yuri_format_to_v4l2(yuri::format_t fmt)
{
	if (formats_map.count(fmt)) return formats_map[fmt];
	return 0;
//		throw exception::Exception("Unknown format");
}
/** Converts v4l2 format to yuri::format_t
 * \param fmt Pixel format as yuri::format_t
 * \return v4l2 pixel format for the specified format.
 */
yuri::format_t v4l2_format_to_yuri(uint32_t fmt)
{
	for (const auto& f: formats_map) {
		if (f.second==fmt) return f.first;
	}
	return core::raw_format::unknown;
	//	case V4L2_PIX_FMT_SN9C20X_I420:	return YURI_FMT_YUV420_PLANAR;
//		throw exception::Exception("Unknown format");
}

std::string get_short_yuri_fmt_name(format_t fmt)
{
	try {
		const auto& fi = core::raw_format::get_format_info(fmt);
		if (fi.short_names.size() > 0) {
			return fi.short_names[0];
		}
	} catch (...) {}
	try {
		const auto& fi = core::compressed_frame::get_format_info(fmt);
		if (fi.short_names.size() > 0) {
			return fi.short_names[0];
		}
	} catch (...) {}
	return {};
}

std::string get_long_yuri_fmt_name(format_t fmt)
{
	try {
		const auto& fi = core::raw_format::get_format_info(fmt);
		return fi.name;
	} catch (...) {}
	try {
		const auto& fi = core::compressed_frame::get_format_info(fmt);
		return fi.name;
	} catch (...) {}
	return {};
}

//std::string get_yuri_fmt_name_from_pixel_format(uint32_t format)
//{
//	auto f = v4l2_format_to_yuri(format);
//	return get_long_yuri_fmt_name(f);
//}

}
