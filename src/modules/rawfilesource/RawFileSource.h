/*!
 * @file 		RawFileSource.h
 * @author 		Zdenek Travnicek
 * @date 		31.3.2011
 * @date		16.3.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef RAWFILESOURCE_H_
#define RAWFILESOURCE_H_

#include "yuri/core/BasicIOThread.h"
//#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace rawfilesource {


//using namespace boost::posix_time;

class RawFileSource: public core::BasicIOThread
{
public:
	RawFileSource(log::Log &_log, core::pwThreadBase parent,core::Parameters &parameters);
	virtual ~RawFileSource();
	static core::pBasicIOThread generate(log::Log &_log,core::pwThreadBase parent,
			core::Parameters& parameters);
	static core::pParameters configure();
	void run();
protected:
	virtual bool set_param(const core::Parameter &parameter);
	bool read_chunk();
	std::string next_file();
	core::pBasicFrame frame;
	yuri::size_t position, chunk_size, width, height;
	yuri::format_t output_format;
	double fps;
	std::string path;
	time_value last_send;
	std::ifstream file;
	bool keep_alive,loop, failed_read, sequence;
	yuri::usize_t block;
	yuri::size_t loop_number;
	size_t sequence_pos;

};

}

}

#endif /* RAWFILESOURCE_H_ */
