/*!
 * @file 		PngEncoder.cpp
 * @author 		<Your name>
 * @date		02.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "PngEncoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include <png.h>

namespace yuri {
namespace png {


IOTHREAD_GENERATOR(PngEncoder)

core::Parameters PngEncoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::RawVideoFrame>::configure();
	p.set_description("PngEncoder");
	return p;
}

namespace {
void write_data(png_structp png_ptr, png_bytep data_in, png_size_t length)
{
	uvector<uint8_t>* data = reinterpret_cast<uvector<uint8_t>*>(png_get_io_ptr(png_ptr));
//	printf("Writing %lu bytes\n",length);
	data->insert(data->end(), data_in, data_in+length);
}
void flush_data(png_structp)
{

}
void report_error(png_structp png_ptr, png_const_charp msg)
{
	log::Log& log = *reinterpret_cast<log::Log*>(png_get_error_ptr(png_ptr));
	log[log::error] << msg;
	throw std::runtime_error(std::string("Failed to encode PNG file: ")+msg);
}
void report_warning(png_structp png_ptr, png_const_charp msg)
{
	log::Log& log = *reinterpret_cast<log::Log*>(png_get_error_ptr(png_ptr));
	log[log::warning] << msg;
}

}
PngEncoder::PngEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("png_encoder"))
{
	IOTHREAD_INIT(parameters)
}

PngEncoder::~PngEncoder() noexcept
{
}

core::pFrame PngEncoder::do_special_single_step(const core::pRawVideoFrame& frame)
{
	format_t input_format = frame->get_format();
	using namespace core::raw_format;
	bool bgr = false;
//	bool alpha_start = false;
	int png_format = 0;
	size_t depth = 8;
	size_t bpp = 0;
	switch(input_format) {
		case y8: png_format = PNG_COLOR_TYPE_GRAY; bpp = 8; break;
		case y16: png_format = PNG_COLOR_TYPE_GRAY; depth = 16; bpp = 16; break;
		case rgb24: png_format = PNG_COLOR_TYPE_RGB; bpp = 24; break;
		case rgb48: png_format = PNG_COLOR_TYPE_RGB; depth = 16; bpp = 48; break;
		case bgr24: png_format = PNG_COLOR_TYPE_RGB; bgr = true; bpp = 24; break;
		case bgr48: png_format = PNG_COLOR_TYPE_RGB; depth = 16; bgr = true; bpp = 48; break;

		case rgba32: png_format = PNG_COLOR_TYPE_RGBA; bpp = 32; break;
		case rgba64: png_format = PNG_COLOR_TYPE_RGBA; depth = 16; bpp = 64; break;
		case bgra32: png_format = PNG_COLOR_TYPE_RGBA; bgr = true; bpp = 32; break;
		case bgra64: png_format = PNG_COLOR_TYPE_RGBA; depth = 16; bgr = true; bpp = 64; break;

		// These formats are commented out as I haven't figured, how to tell PNG the A component is first...

//		case argb32: png_format = PNG_COLOR_TYPE_RGBA; bpp = 32; alpha_start = true; break;
//		case argb64: png_format = PNG_COLOR_TYPE_RGBA; depth = 16; bpp = 64; alpha_start = true; break;
//		case abgr32: png_format = PNG_COLOR_TYPE_RGBA; bgr = true; bpp = 32; alpha_start = true; break;
//		case abgr64: png_format = PNG_COLOR_TYPE_RGBA; depth = 16; bgr = true; bpp = 64; alpha_start = true; break;
	}
	if (bpp == 0) {
		log[log::warning] << "Unsupported format! (" << core::raw_format::get_format_name(input_format) << ")";
		return {};
	}

	png_infop info_ptr = nullptr;
	unique_ptr<png_struct, function<void(png_structp)>> png_ptrx (png_create_write_struct(PNG_LIBPNG_VER_STRING, &log, report_error, report_warning),
			[&info_ptr](png_structp p){
				if (p) png_destroy_write_struct(&p, &info_ptr);
			});

	png_structp png_ptr = png_ptrx.get();
	if (!png_ptr) {
		log[log::warning] << "Failed to initialize png read";
		return {};
	}
	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		log[log::warning] << "Failed to initialize png info";
		return {};
	}

	try {
		uvector<uint8_t> data;
		data.reserve(10485760);
		png_set_write_fn(png_ptr, &data, write_data, flush_data);

		resolution_t res = frame->get_resolution();
		png_set_IHDR(png_ptr, info_ptr, res.width, res.height, depth, png_format,
				PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
				PNG_FILTER_TYPE_DEFAULT);
		if (bgr) png_set_bgr(png_ptr);

		png_write_info(png_ptr, info_ptr);
		std::vector<png_bytep> rows(res.height);
		size_t linesize = res.width * bpp / 8;
		png_bytep out_data = PLANE_RAW_DATA(frame,0);
		for (dimension_t i = 0; i < res.height; ++i) {
			rows[i] = out_data;
			out_data += linesize;
		}
		if (std::distance(PLANE_RAW_DATA(frame,0),out_data) > static_cast<ptrdiff_t>(PLANE_SIZE(frame,0))) {
			log[log::error] << "Providing libpng with " << std::distance(PLANE_RAW_DATA(frame,0),out_data) << " bytes, when only " <<  PLANE_SIZE(frame,0) << "was available...";
			return {};
		}
		png_write_image(png_ptr, rows.data());
		png_write_end(png_ptr, info_ptr);

		// TODO: This copies the data, it would be better to provide a way to just move them in...
		core::pCompressedVideoFrame frame_out = core::CompressedVideoFrame::create_empty(
				core::compressed_frame::png, res, data.data(), data.size());

		return frame_out;
	}
	catch (std::runtime_error&) {}
	return {};
}
core::pFrame PngEncoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	if(target_format != core::compressed_frame::png) return {};
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(input_frame);
	if (!frame) return {};
	return do_special_single_step(frame);
}
bool PngEncoder::set_param(const core::Parameter& param)
{
	return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(param);
}

} /* namespace png */
} /* namespace yuri */
