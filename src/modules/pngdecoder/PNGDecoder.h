/*
 * PNGDecode.h
 *
 *  Created on: Jul 27, 2009
 *      Author: neneko
 */

#ifndef PNGDECODE_H_
#define PNGDECODE_H_
#include <yuri/io/BasicIOThread.h>
#include <yuri/config/Config.h>
#include <png.h>
namespace yuri {

namespace io {
using yuri::log::Log;
using namespace yuri::config;
using namespace std;

class PNGDecoder: public BasicIOThread {
public:
	PNGDecoder(Log &_log, pThreadBase parent, Parameters& parameters);
	virtual ~PNGDecoder();
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual bool step();
	static bool validatePng(pBasicFrame f);
protected:
	static void readData(png_structp pngPtr, png_bytep data, png_size_t length);
	void readData(png_bytep data, png_size_t length);
	long position;
	pBasicFrame f;
	//png_structp pngPtr;
	//png_infop infoPtr;
};

}

}

#endif /* PNGDECODE_H_ */
