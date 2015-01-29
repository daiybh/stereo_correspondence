/*!
 * @file 		JpegDecoder.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		31.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "JpegDecoder.h"
#include "jpeg_common.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace jpeg {


IOTHREAD_GENERATOR(JpegDecoder)


core::Parameters JpegDecoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::CompressedVideoFrame>::configure();
	p.set_description("JpegDecoder");
	p["format"]["Output format"]="RGB24";
	p["fast"]["Faster decoding with slightly worse quality"]=false;
	return p;
}

namespace {

void error_exit(jpeg_common_struct* /*cinfo*/)
{
	throw std::runtime_error("Error");
}
}


JpegDecoder::JpegDecoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
:core::SpecializedIOFilter<core::CompressedVideoFrame>(log_, parent, std::string("jpeg_decoder")),
 fast_(false),output_format_(core::raw_format::rgb24)
{
	IOTHREAD_INIT(parameters)
}

JpegDecoder::~JpegDecoder() noexcept
{
}

core::pFrame JpegDecoder::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	// TODO SET FORMAT
	output_format_ = target_format;
	core::pCompressedVideoFrame frame = dynamic_pointer_cast<core::CompressedVideoFrame>(input_frame);
	if (!frame) return {};
	return do_special_single_step(frame);
}

core::pFrame JpegDecoder::do_special_single_step(const core::pCompressedVideoFrame& frame)
{
	format_t fmt = frame->get_format();
	if ((fmt != core::compressed_frame::jpeg) &&
			(fmt != core::compressed_frame::mjpg))	{
		log[log::info] << "Unsupported format!";
		return {};
	}

	unique_ptr<jpeg_decompress_struct, function<void(jpeg_decompress_struct*)>> cinfox
			(new jpeg_decompress_struct,[](jpeg_decompress_struct* cinfo){
		jpeg_destroy_decompress(cinfo);
		delete cinfo;
	});

	// Pointer to decompress info owned by cinfox
	// It' not neccessary, but it makes the code more eclipse friendly and it costs nothing.
	jpeg_decompress_struct* const cinfo = cinfox.get();
	try{
		cinfo->client_data=reinterpret_cast<void*>(this);
		struct jpeg_error_mgr jerr;
		cinfo->err = jpeg_std_error(&jerr);
		cinfo->err->error_exit=error_exit;

		jpeg_create_decompress(cinfo);

		jpeg_mem_src(cinfo, frame->data(), frame->size());


		if (jpeg_read_header(cinfo,true)!=JPEG_HEADER_OK) {
			log[log::warning] << "Unrecognized file header!!";
			return {};
		}

		cinfo->out_color_space = yuri_to_jpeg(output_format_);
		if (cinfo->out_color_space == JCS_UNKNOWN) {
			log[log::error] << "Unsupported color space";
			return {};
		}
		cinfo->dct_method = JDCT_FLOAT;
		if (fast_) {
			cinfo->do_fancy_upsampling= false;
			cinfo->do_block_smoothing = false;
		}
		jpeg_start_decompress(cinfo);


		resolution_t res = {cinfo->image_width, cinfo->image_height};
		core::pRawVideoFrame out_frame = core::RawVideoFrame::create_empty(output_format_, res);
		size_t out_linesize = PLANE_DATA(out_frame, 0).get_line_size();
		size_t planes = 1; // Only 1 plane supported

		std::vector<std::vector<JSAMPROW> > row_pointers(planes);
		for (size_t p=0;p<planes;++p) {
			size_t plane_height = res.height * cinfo->comp_info[p].h_samp_factor/cinfo->comp_info[0].h_samp_factor;
			row_pointers[p].resize(plane_height);
			int ls = out_linesize*cinfo->comp_info[p].v_samp_factor/cinfo->comp_info[0].v_samp_factor;
			for (size_t h = 0; h < plane_height; ++h) {
				row_pointers[p][h]=PLANE_RAW_DATA(out_frame,p) + h*ls;
			}
		}
		std::vector<JSAMPARRAY> arrays_pointers(planes);
		for (size_t p=0;p<planes;++p) {
			arrays_pointers[p]=&(row_pointers[p][0]);
		}
		yuri::size_t processed=0;

		while (cinfo->output_scanline < cinfo->image_height) {
			/*if (!raw_)*/ processed = jpeg_read_scanlines(cinfo, &row_pointers[0][cinfo->output_scanline], res.height - cinfo->output_scanline);
	//		else {
	//			log[log::info] << "processing raw line " << cinfo.output_scanline;
	//			//JSAMPARRAY arr_pointer = &row_pointers[0];
	//			for (int p=0;p<planes;++p) {
	//				size_t off = cinfo.output_scanline*cinfo.comp_info[p].v_samp_factor/cinfo.comp_info[0].v_samp_factor;
	//				arrays_pointers[p]=&row_pointers[p][off];
	//				log[log::info] << p<< " Setting ptr at offset " << std::distance(row_pointers[p][0],row_pointers[p][off]);
	//			}
	//			JSAMPIMAGE img = &arrays_pointers[0];
	//			processed = jpeg_read_raw_data(&cinfo, img, height-cinfo.output_scanline);
	//			log[log::info] << "processed " << processed;
	//		}

			if (!processed) {
				log[log::error] << "No lines processed ... corrupted file?";
				return {};
			}
		}

		jpeg_finish_decompress(cinfo);
		return out_frame;
	}
	catch (std::runtime_error& ) {
		log[log::warning] << "Decoding failed";
	}
	return {};
}

bool JpegDecoder::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(output_format_, "format", [](const core::Parameter&p){ return core::raw_format::parse_format(p.get<std::string>()); })
			(fast_, "fast"))
		return true;
	return core::SpecializedIOFilter<core::CompressedVideoFrame>::set_param(param);
}

} /* namespace jpeg */
} /* namespace yuri */
