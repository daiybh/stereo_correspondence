/*!
 * @file 		JPEGDecoder.h
 * @author 		Zdenek Travnicek
 * @date 		3.8.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
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
using namespace yuri::config;

class JPEGDecoder:public BasicIOThread {
public:
	JPEGDecoder(Log &_log, pThreadBase parent);
	virtual ~JPEGDecoder();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();

	static bool validate(pBasicFrame f);
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
	pBasicFrame frame;
	int line_width_mult;
	bool aborted;

};

}

}
#endif /* JPEGDECODER_H_ */
