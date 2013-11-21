/*!
 * @file 		JpegEncoder.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		31.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "JpegEncoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "jpeg_common.h"
namespace yuri {
namespace jpeg {


IOTHREAD_GENERATOR(JpegEncoder)

core::Parameters JpegEncoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("JpegEncoder");
	p["quality"]["Jpeg quality"]=90;
	return p;
}
namespace {

void error_exit(jpeg_common_struct* /*cinfo*/)
{
	throw std::runtime_error("Error");
}
std::vector<format_t> supported_formats = {
	core::raw_format::rgb24,
	core::raw_format::bgr24,
	core::raw_format::rgba32,
	core::raw_format::bgra32,
	core::raw_format::abgr32,
	core::raw_format::argb32,
	core::raw_format::yuv444,
	core::raw_format::y8,
};
}

JpegEncoder::JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("jpeg_encoder")),
quality_(90)
{
	IOTHREAD_INIT(parameters)
	set_supported_formats(supported_formats);
}

JpegEncoder::~JpegEncoder() noexcept
{
}

core::pFrame JpegEncoder::do_special_single_step(const core::pRawVideoFrame& frame)
{
	unique_ptr<jpeg_compress_struct, function<void(jpeg_compress_struct*)>> cinfox(
			new jpeg_compress_struct, [](jpeg_compress_struct* ptr)
			{
				jpeg_destroy_compress(ptr);
				delete ptr;
			});
	jpeg_compress_struct* cinfo = cinfox.get();
	cinfo->client_data=reinterpret_cast<void*>(this);
	struct jpeg_error_mgr jerr;
	cinfo->err = jpeg_std_error(&jerr);
	cinfo->err->error_exit=error_exit;
	try {
		jpeg_create_compress(cinfo);

		format_t fmt = frame->get_format();
		J_COLOR_SPACE cs = yuri_to_jpeg(fmt);
		if (cs == JCS_UNKNOWN) {
			log[log::warning] << "Unsupported format";
			return {};
		}

		uint8_t *buffer = nullptr;
		unsigned long buffer_size = 0;
		jpeg_mem_dest(cinfo, &buffer, &buffer_size);
		const auto& fi = core::raw_format::get_format_info(fmt);
//		size_t bpp = core::raw_format::get_fmt_bpp(fmt)/8;
		resolution_t res = frame->get_resolution();
		cinfo->image_width = res.width;
		cinfo->image_height = res.height;

		// This is probably not correct for all formats, but it should work for all formats supported here.
		cinfo->input_components = fi.planes[0].components.size();
		cinfo->in_color_space = cs;

		jpeg_set_defaults(cinfo);
		jpeg_set_quality(cinfo, quality_, true);
		jpeg_start_compress(cinfo, true);

		uint8_t* data = PLANE_RAW_DATA(frame,0);
		size_t line_size = PLANE_DATA(frame,0).get_line_size();
		JSAMPROW row_pointer;

		while (cinfo->next_scanline < cinfo->image_height) {
			row_pointer = reinterpret_cast<JSAMPROW>(data+cinfo->next_scanline*line_size);
			jpeg_write_scanlines(cinfo, &row_pointer, 1);
		}
		jpeg_finish_compress(cinfo);

		log[log::verbose_debug] << "Buffer is now " << buffer_size << " bytes long";
		if (buffer_size) return core::CompressedVideoFrame::create_empty(core::compressed_frame::jpeg, res, buffer, buffer_size);

	}
	catch (std::runtime_error& ) {
		log[log::error] << "Failed to encode frame";
	}
	return {};
}
core::pFrame JpegEncoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	if (target_format != core::compressed_frame::jpeg) return {};
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) return {};
	return do_special_single_step(frame);
}
bool JpegEncoder::set_param(const core::Parameter& param)
{
	if (param.get_name() == "quality") {
		quality_ = param.get<size_t>();
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
	return true;
}

} /* namespace jpeg2 */
} /* namespace yuri */
