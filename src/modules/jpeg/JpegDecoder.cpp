/*!
 * @file 		JpegDecoder.cpp
 * @author 		<Your name>
 * @date		31.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "JpegDecoder.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include <jpeglib.h>


namespace yuri {
namespace jpeg {


IOTHREAD_GENERATOR(JpegDecoder)


core::Parameters JpegDecoder::configure()
{
	core::Parameters p = core::SpecializedIOFilter<core::CompressedVideoFrame>::configure();
	p.set_description("JpegDecoder");
	return p;
}

namespace {
void init_source(jpeg_decompress_struct*) {}
int fill_input(jpeg_decompress_struct*) { return 1; }
int resync_data(jpeg_decompress_struct* /*cinfo*/, int /*desired*/) { return 1; }
void term_source(jpeg_decompress_struct* /*cinfo*/) {}

void error_exit(jpeg_common_struct* cinfo)
{
//	JPEGDecoder *dec = reinterpret_cast<JPEGDecoder*>(cinfo->client_data);
//	if (dec) dec->abort();
}
void skip_data(jpeg_decompress_struct* cinfo, long numbytes)
{
	if ((long)(cinfo->src->bytes_in_buffer) < numbytes) cinfo->src->bytes_in_buffer=0;
	else {
		cinfo->src->bytes_in_buffer-=numbytes;
		cinfo->src->next_input_byte+=numbytes;
	}
}
}


JpegDecoder::JpegDecoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters)
:core::SpecializedIOFilter<core::CompressedVideoFrame>(log_, parent, std::string("jpeg_decoder")),
 aborted_(false),output_format_(0)
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


	jpeg_decompress_struct cinfo;
	cinfo.client_data=reinterpret_cast<void*>(this);
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	cinfo.err->error_exit=error_exit;
	jpeg_create_decompress(&cinfo);
	cinfo.src = new jpeg_source_mgr;
	cinfo.src->init_source=init_source;
	cinfo.src->next_input_byte=(JOCTET *)frame->data();
	cinfo.src->bytes_in_buffer=frame->size();
	cinfo.src->fill_input_buffer=fill_input;
	cinfo.src->resync_to_restart=resync_data;
	cinfo.src->skip_input_data=skip_data;
	cinfo.src->term_source=term_source;

	aborted_ = false;

	if (jpeg_read_header(&cinfo,true)!=JPEG_HEADER_OK) {
		log[log::warning] << "Unrecognized file header!!";
		return {};//true;
	}


}
bool JpegDecoder::set_param(const core::Parameter& param)
{
	return core::SpecializedIOFilter<core::CompressedVideoFrame>::set_param(param);
}

} /* namespace jpeg */
} /* namespace yuri */
