/*!
 * @file 		PNGEncoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		27.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "PNGEncoder.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace png {

REGISTER("pngencoder",PNGEncoder)

IO_THREAD_GENERATOR(PNGEncoder)

core::pParameters PNGEncoder::configure()
{
	core::pParameters p = BasicIOThread::configure();
	return p;
}

PNGEncoder::PNGEncoder(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters) IO_THREAD_CONSTRUCTOR:
	core::BasicIOThread(_log, parent, 1, 1,"PNGEncoder"), memSize(0)
{
	IO_THREAD_INIT("PNGEncoder")
}

PNGEncoder::~PNGEncoder()
{

}

bool PNGEncoder::step() {
	png_structp pngPtr = 0;
	png_infop infoPtr = 0;
	int pngcolortype = 0;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;
	log[log::verbose_debug] << "Reading packet " << frame->get_size() << " bytes long" << std::endl;
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
		log[log::error] << "Unknown BPP (colorspace: " << colorspace
				<< ", not processing" << std::endl;
		frame.reset();
		return true;
	}
	if (static_cast<yuri::size_t>(bpp * width * height) != frame->get_size()) {
		log[log::error] << "Wrong frame size!! (got:" << frame->get_size()
				<< ", expected: " << bpp * width * height << ")" << std::endl;
		frame.reset();
		return true;
	}
	boost::posix_time::ptime t1(
			boost::posix_time::microsec_clock::universal_time());
	pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!pngPtr) {
		log[log::error] << "ERROR: Couldn't initialize png read struct" << std::endl;
		frame.reset();
		return true;
	}
	infoPtr = png_create_info_struct(pngPtr);
	if (!infoPtr) {
		log[log::error] << "ERROR: Couldn't initialize png info struct" << std::endl;
		png_destroy_write_struct(&pngPtr, (png_infopp) 0);
		frame.reset();
		return true;
	}

	png_set_write_fn(pngPtr, (void*) this, PNGEncoder::writeData,
			PNGEncoder::flushData);

	if (memSize != width * height * bpp) {
		memSize = width * height * bpp;
		memory.resize(memSize);
	}
	position = 0;
	png_set_IHDR(pngPtr, infoPtr, width, height, 8, pngcolortype,
			PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
			PNG_FILTER_TYPE_DEFAULT);
	png_write_info(pngPtr, infoPtr);
	png_bytep *rows = new png_bytep[height];
	int row_size = width * bpp;
	for (int i = 0; i < height; ++i) {
		rows[i] = reinterpret_cast<png_bytep>(PLANE_RAW_DATA(frame,0) + i * row_size);
	}
	png_write_image(pngPtr, rows);
	png_write_end(pngPtr, infoPtr);
	png_destroy_write_struct(&pngPtr, &infoPtr);
	boost::posix_time::ptime t2(
			boost::posix_time::microsec_clock::universal_time());
	boost::posix_time::time_period tp(t1, t2);
	log[log::debug] << "Compression took: " << tp.length().total_microseconds()
			<< " us" << std::endl;
	if (position || out[0]) {
		core::pBasicFrame out_frame = allocate_frame_from_memory(&memory[0],position);
		push_video_frame(0,out_frame,YURI_IMAGE_PNG,frame->get_width(),frame->get_height());
	}
	return true;
}

void PNGEncoder::writeData(png_structp pngPtr, png_bytep data, png_size_t length) {
	PNGEncoder *png = (PNGEncoder*) png_get_io_ptr(pngPtr);
	png->writeData(data, length);
}
void PNGEncoder::flushData(png_structp /*pngPtr*/) {

}

void PNGEncoder::writeData(png_bytep data, png_size_t length) {
	if (!memory.size())
		return;
	//memcpy(memory.get() + position, data, length);
	std::copy(data, data + length, memory.begin() + position);
	position += length;
}

void PNGEncoder::handleError(png_structp pngPtr, png_const_charp error_msg) {
	PNGEncoder *png = (PNGEncoder*) png_get_io_ptr(pngPtr);
	png->printError(log::error, error_msg);
}
void PNGEncoder::handleWarning(png_structp pngPtr, png_const_charp error_msg) {
	PNGEncoder *png = (PNGEncoder*) png_get_io_ptr(pngPtr);
	png->printError(log::warning, error_msg);
}
void PNGEncoder::printError(int type, const char * msg) {
	log[(yuri::log::debug_flags) type] << msg << std::endl;
}

}

}

// End of file
