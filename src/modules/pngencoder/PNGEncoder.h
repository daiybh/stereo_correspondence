/*
 * PNGEncode.h
 *
 *  Created on: Jul 27, 2009
 *      Author: neneko
 */

#ifndef PNGENCODE_H_
#define PNGENCODE_H_
#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include <png.h>

namespace yuri {

namespace io {
using yuri::log::Log;
using namespace yuri::config;

class PNGEncode: public BasicIOThread {
public:
	PNGEncode(Log &_log, pThreadBase parent, Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~PNGEncode();
	virtual bool step();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();

protected:
	static void writeData(png_structp pngPtr, png_bytep data, png_size_t length);
	static void flushData(png_structp pngPtr);
	void writeData(png_bytep data, png_size_t length);
	static void handleError(png_structp png_ptr, png_const_charp error_msg);
	static void handleWarning(png_structp png_ptr, png_const_charp error_msg);
	void printError(int type, const char * msg);
	long position;
	pBasicFrame frame;
	shared_array<yuri::ubyte_t> memory;
	long memSize;
};

}

}

#endif /* PNGENCODE_H_ */
