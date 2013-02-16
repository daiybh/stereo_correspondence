/*!
 * @file 		VNCClient.h
 * @author 		Zdenek Travnicek
 * @date 		20.12.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef VNCCLIENT_H_
#define VNCCLIENT_H_

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include "yuri/asio/ASIOTCPSocket.h"

namespace yuri {

namespace io {
struct _color_spec {
	yuri::ushort_t max;
	yuri::ubyte_t shift;
};
struct _pixel_format {
	yuri::ushort_t bpp, depth;
	bool big_endian, true_color;
	_color_spec colors[3];
};
enum _receiving_states {
	awaiting_data,
	receiving_rectangles
};
class VNCClient: public yuri::io::BasicIOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();

	VNCClient(Log &log_,pThreadBase parent,Parameters &parameters)
					IO_THREAD_CONSTRUCTOR;
	virtual ~VNCClient();

protected:
	void run();
	bool connect();
	bool handshake();
	bool set_param(Parameter &p);
	bool process_data();
	bool request_rect(yuri::ushort_t x, yuri::ushort_t y, yuri::ushort_t w, yuri::ushort_t h, bool incremental);
	bool enable_continuous();
	bool set_encodings();
	static inline yuri::uint_t get_uint(yuri::ubyte_t *start);
	static inline yuri::ushort_t get_ushort(yuri::ubyte_t *start);
	static inline void store_ushort(yuri::ubyte_t *start, yuri::ushort_t data);
	static inline void store_uint(yuri::ubyte_t *start, yuri::uint_t data);
	inline yuri::size_t get_pixel(yuri::ubyte_t *buf);
	void move_buffer(yuri::ssize_t offset);
	std::string address;
	yuri::ushort_t port;
//	boost::shared_ptr<ASIOAsyncTCPSocket> socket;
	boost::shared_ptr<ASIOTCPSocket> socket;

	plane_t buffer, image;
	yuri::ubyte_t *buffer_pos, *buffer_end;
	yuri::size_t buffer_size, buffer_free, buffer_valid;
	yuri::ushort_t server_major, server_minor;
	yuri::ushort_t width,height;
	_pixel_format pixel_format;
	_receiving_states state;
	yuri::size_t remaining_rectangles;
	boost::posix_time::ptime last_read, now;
};

}

}

#endif /* VNCCLIENT_H_ */
