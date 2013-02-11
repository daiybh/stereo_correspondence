/*
 * RawFileSource.h
 *
 *  Created on: Mar 31, 2011
 *      Author: worker
 */

#ifndef RAWFILESOURCE_H_
#define RAWFILESOURCE_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace io {

using namespace yuri::config;
using boost::shared_array;
using namespace boost::posix_time;

class RawFileSource: public BasicIOThread
{
public:
	RawFileSource(Log &_log, pThreadBase parent, Parameters &parameters);
	virtual ~RawFileSource();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,
			Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
	void run();
protected:
	virtual bool set_param(Parameter &parameter);
	bool read_chunk();
	shared_ptr<BasicFrame> frame;
	yuri::size_t position, chunk_size, width, height;
	yuri::format_t output_format;
	double fps;
	string path;
	ptime last_send;
	ifstream file;
	bool keep_alive,loop, failed_read;
	yuri::usize_t block;
	yuri::size_t loop_number;
};

}

}

#endif /* RAWFILESOURCE_H_ */
