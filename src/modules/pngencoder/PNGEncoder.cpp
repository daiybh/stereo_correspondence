/*
 * PNGEncode.cpp
 *
 *  Created on: Jul 27, 2009
 *      Author: neneko
 */

#include "PNGEncoder.h"

namespace yuri {

namespace io {

REGISTER("pngencoder",PNGEncode)

IO_THREAD_GENERATOR(PNGEncode)

shared_ptr<Parameters> PNGEncode::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	return p;
}

PNGEncode::PNGEncode(Log &_log, pThreadBase parent, Parameters& parameters) IO_THREAD_CONSTRUCTOR:
	BasicIOThread(_log, parent, 1, 1,"PNGEncode"), memSize(0)
{
	IO_THREAD_INIT("PNGEncode")
}

PNGEncode::~PNGEncode()
{

}

bool PNGEncode::step() {
	png_structp pngPtr = 0;
	png_infop infoPtr = 0;
	int pngcolortype = 0;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;
	log[verbose_debug] << "Reading packet " << frame->get_size() << " bytes long" << std::endl;
	int bpp = 0;
	int width = frame->get_width();
	int height = frame->get_height();
	int colorspace = frame->get_format();

	switch (colorspace) {
	case YURI_FMT_RGBA:
		bpp = 4;
		pngcolortype = PNG_COLOR_TYPE_RGBA;
		break;
	case YURI_FMT_RGB:
		bpp = 3;
		pngcolortype = PNG_COLOR_TYPE_RGB;
		break;
	default:
		bpp = 0;
		break;
	}
	if (!bpp) {
		log[error] << "Unknown BPP (colorspace: " << colorspace
				<< ", not processing" << std::endl;
		frame.reset();
		return true;
	}
	if (static_cast<yuri::size_t>(bpp * width * height) != frame->get_size()) {
		log[error] << "Wrong frame size!! (got:" << frame->get_size()
				<< ", expected: " << bpp * width * height << ")" << std::endl;
		frame.reset();
		return true;
	}
	boost::posix_time::ptime t1(
			boost::posix_time::microsec_clock::universal_time());
	pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!pngPtr) {
		log[error] << "ERROR: Couldn't initialize png read struct" << std::endl;
		frame.reset();
		return true;
	}
	infoPtr = png_create_info_struct(pngPtr);
	if (!infoPtr) {
		log[error] << "ERROR: Couldn't initialize png info struct" << std::endl;
		png_destroy_write_struct(&pngPtr, (png_infopp) 0);
		frame.reset();
		return true;
	}

	png_set_write_fn(pngPtr, (void*) this, PNGEncode::writeData,
			PNGEncode::flushData);

	if (!memory || memSize != width * height * bpp) {
		memSize = width * height * bpp;
		memory.reset(new yuri::ubyte_t[memSize]);
	}
	position = 0;
	png_set_IHDR(pngPtr, infoPtr, width, height, 8, pngcolortype,
			PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
			PNG_FILTER_TYPE_DEFAULT);
	png_write_info(pngPtr, infoPtr);
	png_bytep *rows = new png_bytep[height];
	int row_size = width * bpp;
	for (int i = 0; i < height; ++i) {
		rows[i] = (png_bytep) ((char*) ((*frame)[0].data.get()) + i * row_size);
	}
	png_write_image(pngPtr, rows);
	png_write_end(pngPtr, infoPtr);
	png_destroy_write_struct(&pngPtr, &infoPtr);
	boost::posix_time::ptime t2(
			boost::posix_time::microsec_clock::universal_time());
	boost::posix_time::time_period tp(t1, t2);
	log[debug] << "Compression took: " << tp.length().total_microseconds()
			<< " us" << std::endl;
	if (position || out[0]) {
		pBasicFrame out_frame = allocate_frame_from_memory(memory.get(),position);
		push_video_frame(0,out_frame,YURI_IMAGE_PNG,frame->get_width(),frame->get_height());
	}
	return true;
}

void PNGEncode::writeData(png_structp pngPtr, png_bytep data, png_size_t length) {
	PNGEncode *png = (PNGEncode*) png_get_io_ptr(pngPtr);
	png->writeData(data, length);
}
void PNGEncode::flushData(png_structp /*pngPtr*/) {

}

void PNGEncode::writeData(png_bytep data, png_size_t length) {
	if (!memory)
		return;
	memcpy(memory.get() + position, data, length);
	position += length;
}

void PNGEncode::handleError(png_structp pngPtr, png_const_charp error_msg) {
	PNGEncode *png = (PNGEncode*) png_get_io_ptr(pngPtr);
	png->printError(error, error_msg);
}
void PNGEncode::handleWarning(png_structp pngPtr, png_const_charp error_msg) {
	PNGEncode *png = (PNGEncode*) png_get_io_ptr(pngPtr);
	png->printError(warning, error_msg);
}
void PNGEncode::printError(int type, const char * msg) {
	log[(yuri::log::debug_flags) type] << msg << std::endl;
}

}

}

// End of file
