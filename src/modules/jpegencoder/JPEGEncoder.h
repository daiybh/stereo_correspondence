/*
 * JPEGEncoder.h
 *
 *  Created on: Jul 29, 2009
 *      Author: neneko
 */

#ifndef JPEGENCODER_H_
#define JPEGENCODER_H_

#include <boost/shared_array.hpp>
#include <cstdio>
#include <jpeglib.h>
#include "yuri/config/RegisteredClass.h"

#include "yuri/io/BasicIOThread.h"

namespace yuri {

namespace io {
using yuri::log::Log;
using namespace std;
using namespace yuri::config;
using boost::shared_array;

class JPEGEncoder: public BasicIOThread {
public:
	JPEGEncoder(Log &_log, pThreadBase parent, int level=75, long buffer_size=1048576);
	virtual ~JPEGEncoder();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,
			Parameters& parameters) throw (Exception);
//	static bool configure_converter(Parameters& parameters,
//			long format_in, long format_out) throw (Exception);
	static shared_ptr<Parameters> configure();
protected:
	virtual bool step();
	void setDestManager(jpeg_compress_struct* cinfo);
	void initDestination(j_compress_ptr cinfo);
	int emptyBuffer(j_compress_ptr cinfo);
static	void sInitDestination(j_compress_ptr cinfo);
static	int sEmptyBuffer(j_compress_ptr cinfo);
static	void sTermDestination(j_compress_ptr cinfo);
	yuri::size_t dumpData();
protected:
	shared_ptr<BasicFrame> frame;
	stringstream temp_data;
	int level;
	shared_array<char> buffer;
	long buffer_size, width, height;
};

}

}
#endif /* JPEGENCODER_H_ */
