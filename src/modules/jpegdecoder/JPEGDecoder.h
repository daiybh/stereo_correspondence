/*
 * JPEGDecoder.h
 *
 *  Created on: Aug 3, 2009
 *      Author: neneko
 */

#ifndef JPEGDECODER_H_
#define JPEGDECODER_H_


#include <yuri/io/BasicIOThread.h>
#include <yuri/config/Config.h>
#include <jpeglib.h>
#include "yuri/config/RegisteredClass.h"

namespace yuri {

namespace io {
using yuri::log::Log;
using namespace std;
using namespace yuri::config;

class JPEGDecoder:public BasicIOThread {
public:
	JPEGDecoder(Log &_log, pThreadBase parent);
	virtual ~JPEGDecoder();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();

	static bool validate(shared_ptr<BasicFrame> f);
	virtual bool step();
	void forceLineWidthMult(int mult) { if (mult>1)line_width_mult = mult; else mult=1; }
protected:
	void setDestManager(jpeg_decompress_struct* cinfo);
	static void initSrc(jpeg_decompress_struct* cinfo);
	//void initSource(jpeg_decompress_struct* cinfo);

	static int fillInput(jpeg_decompress_struct* cinfo);
	static void skipData(jpeg_decompress_struct* cinfo, long numbytes);
	static int resyncData(jpeg_decompress_struct* cinfo, int desired);
	static void termSource(jpeg_decompress_struct* cinfo);
	static void errorExit(jpeg_common_struct* cinfo);
	void abort();
	shared_ptr<BasicFrame> frame;
	int line_width_mult;
	bool aborted;

};

}

}
#endif /* JPEGDECODER_H_ */
