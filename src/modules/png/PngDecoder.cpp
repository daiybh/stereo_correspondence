/*!
 * @file 		PngDecoder.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		02.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PngDecoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <png.h>

#if PNG_LIBPNG_VER_MAJOR == 1 && PNG_LIBPNG_VER_MINOR >= 6
#define PNG_VERSION_1_6
#endif

namespace yuri {
namespace png {


IOTHREAD_GENERATOR(PngDecoder)


core::Parameters PngDecoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::CompressedVideoFrame>::configure();
	p.set_description("PngDecoder");
	p["format"]["Output format. If not specified, the format of the image will be used"]="";
	return p;
}

namespace {
bool validate_png(const core::pCompressedVideoFrame& f)
{
	if (!f || f->get_size() < 8) return false;
    return (!png_sig_cmp(reinterpret_cast<png_byte*>(f->data()), 0, 8));
}
struct mem_buffer {
	png_bytep data;// = nullptr;
	png_size_t length;// = 0;
};
void read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
	mem_buffer* buf = reinterpret_cast<mem_buffer*>(png_get_io_ptr(png_ptr));
	png_size_t read_bytes = std::min(length, buf-> length);
	std::copy(buf->data, buf->data + read_bytes, data);
	buf->data+=read_bytes;
	buf->length-=read_bytes;
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

PngDecoder::PngDecoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::CompressedVideoFrame>(log_,parent,std::string("png_decoder")),
ConverterThread(),
requested_format_(0)
{
	IOTHREAD_INIT(parameters)
}

PngDecoder::~PngDecoder() noexcept
{
}
core::pFrame PngDecoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	requested_format_ = target_format;
	core::pCompressedVideoFrame frame = dynamic_pointer_cast<core::CompressedVideoFrame>(input_frame);
	if (frame) return do_special_single_step(frame);
	return {};
}
core::pFrame PngDecoder::do_special_single_step(const core::pCompressedVideoFrame& frame)
{
	if (frame->get_format() != core::compressed_frame::png) return {};

	if (!validate_png(frame)) {
		log[log::error] << "Received frame is not a valid PNG image";
		return {};
	}
	png_infop info_ptr = nullptr;
	unique_ptr<png_struct, function<void(png_structp)>> png_ptrx(png_create_read_struct(PNG_LIBPNG_VER_STRING, &log, report_error, report_warning),
			[&info_ptr](png_structp ptr){
				if (ptr) png_destroy_read_struct(&ptr, &info_ptr, nullptr);
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
		mem_buffer data_buffer { frame->data(), frame->size()};
		png_set_read_fn(png_ptr,&data_buffer, read_data);
//		png_set_sig_bytes(png_ptr, 8);
		png_read_info(png_ptr, info_ptr);
		resolution_t image_res = { png_get_image_width(png_ptr, info_ptr),
									png_get_image_height(png_ptr, info_ptr)};
		size_t depth = png_get_bit_depth(png_ptr, info_ptr);
		size_t channels = png_get_channels(png_ptr, info_ptr);
		png_byte color_type = png_get_color_type(png_ptr, info_ptr);
		log[log::debug] << "Reading image: " << image_res << ", " << depth
			<< " bpp, " << channels << " channels.\n";
		// We don't want any paletted output
		if (color_type == PNG_COLOR_TYPE_PALETTE) {
			png_set_palette_to_rgb(png_ptr);
		}
		if (color_type == PNG_COLOR_TYPE_GRAY && depth <8) {
			png_set_expand_gray_1_2_4_to_8(png_ptr);
			depth = 8;
		}
		if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
			png_set_tRNS_to_alpha(png_ptr);
		}

		bool bgr = false, alpha_first = false;
		if (requested_format_) {
			const auto& fi = core::raw_format::get_format_info(requested_format_);
			size_t req_depth = fi.planes[0].component_bit_depths[0];
			if (depth == 16 && req_depth == 8) {
				png_set_strip_16(png_ptr);
				depth = 8;
			}
#ifdef PNG_VERSION_1_6
			else if (depth == 8 && req_depth==16) {
				png_set_expand_16(png_ptr);
				depth = 16;
			}
#endif
			size_t req_channels = fi.planes[0].components.size();
			if (req_channels != channels) {
#ifdef PNG_VERSION_1_6
				if (req_channels == 1) {
					png_set_rgb_to_gray(png_ptr, PNG_ERROR_ACTION_WARN, PNG_RGB_TO_GRAY_DEFAULT, PNG_RGB_TO_GRAY_DEFAULT);
				} else
#endif
				if (req_channels == 3) {
					if (channels == 1) {
						png_set_gray_to_rgb(png_ptr);
					} else {
						png_set_strip_alpha(png_ptr);
					}
					if (fi.planes[0].components[0]=='B') {
						png_set_bgr(png_ptr);
						bgr = true;
					}
				} else {
					int flags = PNG_FILLER_AFTER;
					if (fi.planes[0].components[0]=='A') {
						alpha_first = true;
						flags = PNG_FILLER_BEFORE;
					}
					png_set_add_alpha(png_ptr, flags, 0xFF);
					if (channels == 1) {
						png_set_gray_to_rgb(png_ptr);
					}
					if (fi.planes[0].components[0]=='B' || fi.planes[0].components[1]=='B') {
						png_set_bgr(png_ptr);
						bgr = true;
					}
				}
			}
			channels = req_channels;
		}
		format_t output_format = 0;
		using namespace core::raw_format;
		switch (channels) {
			case 1:
				output_format = depth==16?y16:y8;
				break;
			case 3:
				output_format = depth==16?(bgr?bgr48:rgb48):(bgr?bgr24:rgb24);
				break;
			case 4:
				if (alpha_first) {
					output_format = depth==16?(bgr?abgr64:argb64):(bgr?abgr32:argb32);
				} else {
					output_format = depth==16?(bgr?bgra64:rgba64):(bgr?bgra32:rgba32);
				}
				break;
			default:
				output_format = 0;
		}
		if (!output_format) {
			log[log::error] << "Failed to determine output format!";
			return {};
		}

		core::pRawVideoFrame frame_out = core::RawVideoFrame::create_empty(output_format, image_res, true);
		std::vector<png_bytep> rows(image_res.height);
		png_bytep data = PLANE_RAW_DATA(frame_out,0);
		const size_t linesize = PLANE_DATA(frame_out,0).get_line_size();
		for (size_t i=0;i<image_res.height;++i) {
			rows[i]=data;
			data+=linesize;
		}
		png_read_image(png_ptr, rows.data());
		return frame_out;
	}
	catch (std::runtime_error&) {}
	return {};
}
bool PngDecoder::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		std::string f = param.get<std::string>();
		if (!f.empty()) requested_format_ = core::raw_format::parse_format(f);
		else requested_format_ = 0;
	} else return core::SpecializedIOFilter<core::CompressedVideoFrame>::set_param(param);
	return true;
}

} /* namespace png2 */
} /* namespace yuri */
