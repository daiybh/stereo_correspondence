/*
 * SimpleTSDemuxer.h
 *
 *  Created on: Oct 3, 2010
 *      Author: neneko
 */

#ifndef SIMPLETSDEMUXER_H_
#define SIMPLETSDEMUXER_H_

#include "yuri/core/IOThread.h"

namespace yuri {

namespace ts_demux {

class SimpleTSDemuxer: public core::IOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	SimpleTSDemuxer(log::Log &log_, core::pwThreadBase parent, core::Parameters& parameters);
	virtual ~SimpleTSDemuxer();
	bool step();
protected:
	yuri::uint_t program;
	yuri::size_t buffer_size, buffer_position, in_buffer_position, packet_size;
	plane_t buffer, in_buffer;
	std::map<yuri::uint_t,yuri::size_t> seen_pids;
	bool parse_stream, strip_pes, quick_strip;
	int width, height;
	yuri::size_t pts, last_pts;
	bool process_packet(yuri::ubyte_t* data);
	void set_buffer_size(yuri::size_t buf_size);
	void send_buffer();
	void process_headers();
	virtual bool set_param(const core::Parameter &parameter);
	core::pBasicFrame stripped_pes();
	yuri::ssize_t find_sync(uint8_t *data,yuri::size_t len);
};

}

}

#endif /* SIMPLETSDEMUXER_H_ */
