/*!
 * @file 		PNGDecoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		27.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "PNGDecoder.h"
#include "yuri/config/RegisteredClass.h"
namespace yuri {

namespace io {

REGISTER("pngdecoder", PNGDecoder)

IO_THREAD_GENERATOR(PNGDecoder)

shared_ptr<Parameters> PNGDecoder::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	return p;
}

PNGDecoder::PNGDecoder(Log &_log, pThreadBase parent, Parameters& parameters):
	BasicIOThread(_log,parent,1,1,"PNGDecoder")
{
	IO_THREAD_INIT("PNGDecoder")
}

PNGDecoder::~PNGDecoder() {
}

bool PNGDecoder::step()
{
	png_structp pngPtr = 0;
	png_infop infoPtr = 0;
	if (!in[0] || !(f = in[0]->pop_frame())) return true;
	log[debug] << "Reading packet " << f->get_size() << " bytes long" << endl;
	boost::posix_time::ptime t1(boost::posix_time::microsec_clock::universal_time());
	if (!validatePng(f)) {
		f.reset();
		return true;
	}
	log[debug] << "Validated" << endl;
	pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!pngPtr) {
		log[error] << "ERROR: Couldn't initialize png read struct" << std::endl;
		f.reset();
		return true;
	}
	infoPtr = png_create_info_struct(pngPtr);
	if (!infoPtr) {
		log[error] << "ERROR: Couldn't initialize png info struct" << std::endl;
		png_destroy_read_struct(&pngPtr, (png_infopp)0, (png_infopp)0);
		f.reset();
		return true;
	}
	position=8;
	png_set_read_fn(pngPtr,(void*)this, PNGDecoder::readData);
	png_set_sig_bytes(pngPtr, position);
	png_read_info(pngPtr, infoPtr);
	uint width = png_get_image_width(pngPtr, infoPtr);
	uint height = png_get_image_height(pngPtr, infoPtr);
	uint depth = png_get_bit_depth(pngPtr, infoPtr);
	uint channels = png_get_channels(pngPtr, infoPtr);
	//uint color = png_get_color_type(pngPtr, infoPtr);
	log[debug] << "Reading image: " << width	<< "x" << height << ", " << depth
		<< " bpp, " << channels << " channels." << endl;
	if (png_get_valid(pngPtr, infoPtr, PNG_INFO_tRNS))
	png_set_tRNS_to_alpha(pngPtr);
	if (depth == 16) {
		png_set_strip_16(pngPtr);
		depth = 8;
	}
	yuri::format_t colorspace;
	if ((depth == 8) && (channels == 4)) colorspace=YURI_FMT_RGBA;
	else if ((depth == 8) && (channels == 3)) colorspace=YURI_FMT_RGB;
	else colorspace=YURI_FMT_NONE;
	if (colorspace!=YURI_FMT_NONE) {
		std::vector<png_bytep> rows(height);
		int row_size = width*depth*channels/8;
		pBasicFrame outframe = allocate_empty_frame(colorspace, width, height, true);
		yuri::ubyte_t *mem=PLANE_RAW_DATA(outframe,0);
		for (int i = 0; i< (int)height; ++i) rows[i]=(png_bytep)(mem+i*row_size);
		png_read_image(pngPtr, &rows[0]);
		push_raw_video_frame(0,outframe);
	}
	png_destroy_read_struct(&pngPtr, &infoPtr,(png_infopp)0);
	boost::posix_time::ptime t2(boost::posix_time::microsec_clock::universal_time());
	boost::posix_time::time_period tp(t1,t2);
	log[debug] << "Decompression took: " << tp.length().total_microseconds()
			<< " us" << endl;
	f.reset();

	return true;
}

bool PNGDecoder::validatePng(pBasicFrame f)
{
	if (!f || f->get_size() < 8) return false;
    return (!png_sig_cmp((png_byte*)(PLANE_RAW_DATA(f,0)), 0, 8));
}

void PNGDecoder::readData(png_structp pngPtr, png_bytep data, png_size_t length)
{
	PNGDecoder *png = (PNGDecoder*)png_get_io_ptr(pngPtr);
	png->readData(data,length);
}
void PNGDecoder::readData(png_bytep data, png_size_t length)
{

	if (!f) return;
	if (static_cast<yuri::size_t>(position) > f->get_size()) return;
	if ((f->get_size() - position) < length) length=f->get_size() - position;
	log[verbose_debug] << "Reading " << length << " bytes from position " << position
		<< std::endl;
	memcpy(data,(char*)(PLANE_RAW_DATA(f,0))+position,length);
	position+=length;
}

}

}
