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
#include "yuri/core/utils/assign_events.h"
#include "jpeg_common.h"
namespace yuri {
namespace jpeg {


IOTHREAD_GENERATOR(JpegEncoder)

core::Parameters JpegEncoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("JpegEncoder");
	p["quality"]["Jpeg quality"]=90;
	p["force_mjpeg"]["Force MJPEG format"]=false;
	return p;
}
namespace {

void error_exit(jpeg_common_struct* /*cinfo*/)
{
	throw std::runtime_error("Error");
}

}

JpegEncoder::JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("jpeg_encoder")),
BasicEventConsumer(log),
quality_(90),force_mjpeg_(false)
{
	IOTHREAD_INIT(parameters)
    log[log::info] << "sf: " << get_jpeg_supported_formats().size();
	set_supported_formats(get_jpeg_supported_formats());
}

JpegEncoder::~JpegEncoder() noexcept
{
}

core::pFrame JpegEncoder::do_special_single_step(core::pRawVideoFrame frame)
{
	process_events();
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
		cinfo->image_width = static_cast<JDIMENSION>(res.width);
		cinfo->image_height = static_cast<JDIMENSION>(res.height);

		// This is probably not correct for all formats, but it should work for all formats supported here.
		cinfo->input_components = static_cast<int>(fi.planes[0].components.size());
		cinfo->in_color_space = cs;

		jpeg_set_defaults(cinfo);
		jpeg_set_quality(cinfo, static_cast<int>(quality_), true);
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
		auto out_fmt = force_mjpeg_?core::compressed_frame::mjpg:core::compressed_frame::jpeg;
		if (buffer_size) return core::CompressedVideoFrame::create_empty(out_fmt, res, buffer, buffer_size);

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
	if (assign_parameters(param)
			(quality_, "quality")
			(force_mjpeg_, "force_mjpeg"))
		return true;
	return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
}

bool JpegEncoder::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			.ranged(quality_, 0, 100, "quality")
			(force_mjpeg_, "force_mjpeg"))
		return true;
	return false;
}

} /* namespace jpeg2 */
} /* namespace yuri */
