/*!
 * @file 		PNGEncoder.h
 * @author 		Zdenek Travnicek
 * @date 		27.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef PNGENCODE_H_
#define PNGENCODE_H_
#include "yuri/core/IOThread.h"
#include <png.h>

namespace yuri {

namespace png {

class PNGEncoder: public core::IOThread {
public:
	PNGEncoder(log::Log &_log, core::pwThreadBase parent,core::Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~PNGEncoder();
	virtual bool step();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

protected:
	static void writeData(png_structp pngPtr, png_bytep data, png_size_t length);
	static void flushData(png_structp pngPtr);
	void writeData(png_bytep data, png_size_t length);
	static void handleError(png_structp png_ptr, png_const_charp error_msg);
	static void handleWarning(png_structp png_ptr, png_const_charp error_msg);
	void printError(int type, const char * msg);
	long position;
	core::pBasicFrame frame;
	std::vector<yuri::ubyte_t> memory;
	long memSize;
};

}

}

#endif /* PNGENCODE_H_ */
