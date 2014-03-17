/*!
 * @file 		RawFileSource.h
 * @author 		Zdenek Travnicek
 * @date 		31.3.2011
 * @date		16.3.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2011 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef RAWFILESOURCE_H_
#define RAWFILESOURCE_H_

#include "yuri/core/thread/IOThread.h"
//#include <boost/date_time/posix_time/posix_time.hpp>

namespace yuri {

namespace rawfilesource {


enum class frame_type_t {
	unknown,
	raw_video,
	compressed_viceo,
	raw_audio
};

class RawFileSource: public core::IOThread
{
	using base_type = core::IOThread;
public:
	RawFileSource(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~RawFileSource() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	void run();
protected:
	virtual bool set_param(const core::Parameter &parameter);
	bool read_chunk();
	std::string next_file();
	core::pFrame frame;
	yuri::size_t position, chunk_size, width, height;
	yuri::format_t output_format;
	double fps;
	std::string path;
	timestamp_t last_send;
	std::ifstream file;
	bool keep_alive,loop, failed_read, sequence;
	size_t block;
	yuri::size_t loop_number;
	size_t sequence_pos;

	frame_type_t frame_type_;
};

}

}

#endif /* RAWFILESOURCE_H_ */
