/*!
 * @file 		JPEGDecoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		3.8.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "JPEGDecoder.h"
#include "yuri/core/Module.h"

namespace yuri {

namespace jpg {

REGISTER("jpegdecoder",JPEGDecoder)

core::pBasicIOThread JPEGDecoder::generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters)
{
	shared_ptr<JPEGDecoder> j( new JPEGDecoder(_log,parent));
	j->forceLineWidthMult(parameters["line_multiply"].get<int>());
	return j;
}
core::pParameters JPEGDecoder::configure()
{
	core::pParameters p = BasicIOThread::configure();
	(*p)["line_multiply"]=1;
	return p;
}


JPEGDecoder::JPEGDecoder(log::Log &_log, core::pwThreadBase parent)
	:BasicIOThread(_log,parent,1,1),line_width_mult(1),aborted(false)
{
	log.setLabel("[JPEGDecoder]");
}

JPEGDecoder::~JPEGDecoder() {
}

bool JPEGDecoder::step() {
	int width, height, colorspace;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;
	bool decompressed = false;
	aborted = false;
	//log[log::debug] << "Reading packet " << frame->get_size() << " bytes long" << std::endl;

	struct jpeg_decompress_struct cinfo;
	cinfo.client_data=reinterpret_cast<void*>(this);
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	cinfo.err->error_exit=errorExit;
	jpeg_create_decompress(&cinfo);
	setDestManager(&cinfo);


	int bpp;
	int linesize;
	shared_array<yuri::ubyte_t> mem;
	core::pBasicFrame out_frame (new core::BasicFrame(1));
	//while (cinfo.next_scanline < cinfo.image_height) {
	try {
		if (jpeg_read_header(&cinfo,true)!=JPEG_HEADER_OK) {
			log[log::warning] << "Unrecognized file header!!" << std::endl;
			return true;
		}
		jpeg_start_decompress(&cinfo);
		if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		width = cinfo.image_width;
		if (line_width_mult > 1 && (width % line_width_mult))
			width = width + line_width_mult - (width % line_width_mult);
		height = cinfo.image_height;
		bpp = cinfo.output_components;
		linesize = width*bpp;
		colorspace= (bpp==3?YURI_FMT_RGB24:bpp==4?YURI_FMT_RGB32:YURI_FMT_NONE);
		log[log::verbose_debug] << "Allocating " << linesize * height << " bytes for image "
			<< width << "x" << height << " @ " << 8*bpp << "bpp" << std::endl;
		PLANE_DATA(out_frame,0).resize(linesize * height);
//		mem=allocate_memory_block(linesize * height);
//		(*out_frame)[0].set(mem,linesize * height);
		JSAMPROW row_pointer;

		yuri::size_t completed = 0, processed=0;
		for (int i=0;i<height;++i) {
			row_pointer = (JSAMPROW) PLANE_RAW_DATA(out_frame,0) + i*linesize;
			processed = jpeg_read_scanlines(&cinfo, &row_pointer, 1);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
			completed += processed;
			if (static_cast<int>(completed) >= height) break;
			if (!processed) {
				log[log::error] << "No lines processed ... corrupt file?" << std::endl;
				break;
			}
		}
		if (processed) {
			jpeg_finish_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
			decompressed = true;
			jpeg_destroy_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		} else {
			decompressed = false;
			jpeg_destroy_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		}
	}
	catch (yuri::exception::Exception &e) {
		log[log::error] << "Decoding failed!: " << e.what() << std::endl;
		jpeg_abort(reinterpret_cast<j_common_ptr>(&cinfo));
		return true;
	}


	if (decompressed && out[0] && out_frame) {
		push_video_frame(0,out_frame,colorspace,width,height);
		out_frame.reset();
		//out[0]->set_params(width,height,colorspace);
		//out[0]->push_frame(mem,linesize*height,true);
	}
	return true;
}


void JPEGDecoder::setDestManager(jpeg_decompress_struct* cinfo)
{
	cinfo->src = new jpeg_source_mgr;
	cinfo->src->init_source=initSrc;
	cinfo->src->next_input_byte=(JOCTET *)PLANE_RAW_DATA(frame,0);
	cinfo->src->bytes_in_buffer=PLANE_SIZE(frame,0);
	cinfo->src->fill_input_buffer=fillInput;
	cinfo->src->resync_to_restart=resyncData;
	cinfo->src->skip_input_data=skipData;
	cinfo->src->term_source=termSource;

}

/// Check if there's valid JPEG magic
bool JPEGDecoder::validate(core::pBasicFrame frame)
{
	if (!frame || PLANE_SIZE(frame,0)<4) return false;
	uint8_t *magic = reinterpret_cast<uint8_t*>(PLANE_RAW_DATA(frame,0));
	if  (magic[0] == 0xff &&
		 magic[1] == 0xd8 ) return true;
		/*(magic[2] == 0xff) || (magic[2] == 0x00))
			return true;*/
//	cout << "Magic: " << hex << (uint)magic[0] << (uint)magic[1] << dec << std::endl;

	return false;
}

void JPEGDecoder::initSrc(jpeg_decompress_struct* /*cinfo*/)
{
}

int JPEGDecoder::fillInput(jpeg_decompress_struct* /*cinfo*/)
{
	return 1;
}
void JPEGDecoder::skipData(jpeg_decompress_struct* cinfo, long numbytes)
{
	if ((long)(cinfo->src->bytes_in_buffer) < numbytes) cinfo->src->bytes_in_buffer=0;
	else {
		cinfo->src->bytes_in_buffer-=numbytes;
		cinfo->src->next_input_byte+=numbytes;
	}
}
int JPEGDecoder::resyncData(jpeg_decompress_struct* /*cinfo*/, int /*desired*/)
{
// Guessing return value...
	return 1;
}
void JPEGDecoder::termSource(jpeg_decompress_struct* /*cinfo*/)
{
}

void JPEGDecoder::errorExit(jpeg_common_struct* cinfo)
{
	JPEGDecoder *dec = reinterpret_cast<JPEGDecoder*>(cinfo->client_data);
	if (dec) dec->abort();
}
void JPEGDecoder::abort()
{
	aborted =  true;
}
}

}
