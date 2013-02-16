/*!
 * @file 		RawFileSource.h
 * @author 		Zdenek Travnicek
 * @date 		31.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef RAWFILESOURCE_H_
#define RAWFILESOURCE_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace io {

using namespace yuri::config;
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
	pBasicFrame frame;
	yuri::size_t position, chunk_size, width, height;
	yuri::format_t output_format;
	double fps;
std::string path;
	ptime last_send;
	std::ifstream file;
	bool keep_alive,loop, failed_read;
	yuri::usize_t block;
	yuri::size_t loop_number;
};

}

}

#endif /* RAWFILESOURCE_H_ */
