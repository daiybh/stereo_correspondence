/*!
 * @file 		VNCClient.h
 * @author 		Zdenek Travnicek
 * @date 		20.12.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2011 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef VNCCLIENT_H_
#define VNCCLIENT_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/socket/StreamSocket.h"
#include "yuri/core/utils/uvector.h"
namespace yuri {

namespace vnc {
struct _color_spec {
	uint16_t max;
	uint8_t shift;
};
struct _pixel_format {
	uint16_t bpp, depth;
	bool big_endian, true_color;
	_color_spec colors[3];
};
enum _receiving_states {
	awaiting_data,
	receiving_rectangles
};
class VNCClient: public yuri::core::IOThread {
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();

	VNCClient(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~VNCClient() noexcept;

private:
	virtual void run() override;
	bool connect();
	bool handshake();
	bool set_param(const core::Parameter &p);
	bool process_data();
	bool request_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, bool incremental);
	bool enable_continuous();
	bool set_encodings();
	static inline uint32_t get_uint(uint8_t *start);
	static inline uint16_t get_ushort(uint8_t *start);
	static inline void store_ushort(uint8_t *start, uint16_t data);
	static inline void store_uint(uint8_t *start, uint32_t data);
	inline yuri::size_t get_pixel(uint8_t *buf);
	void move_buffer(yuri::ssize_t offset);

	size_t read_data_at_least(uint8_t* data, size_t size, size_t at_least);
	std::string address;
	uint16_t port;
//	boost::shared_ptr<ASIOAsyncTCPSocket> socket;
	//boost::shared_ptr<asio::ASIOTCPSocket> socket;
	core::socket::pStreamSocket socket_;

	uvector<uint8_t> buffer, image;
	uint8_t *buffer_pos, *buffer_end;
	yuri::size_t buffer_size, buffer_free, buffer_valid;
	uint16_t server_major, server_minor;
	uint16_t width,height;
	_pixel_format pixel_format;
	_receiving_states state;
	yuri::size_t remaining_rectangles;
//	boost::posix_time::ptime last_read, now;
	timestamp_t last_read;
};

}

}

#endif /* VNCCLIENT_H_ */
