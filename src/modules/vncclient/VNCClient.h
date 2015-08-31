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
struct color_spec_t {
	uint16_t max;
	uint8_t shift;
};
struct pixel_format_t {
	uint16_t bpp, depth;
	bool big_endian, true_color;
	color_spec_t colors[3];
};
enum receiving_states_t {
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
	bool set_param(const core::Parameter &p) override;
	bool process_data();
	bool request_rect(geometry_t geometry, bool incremental);
	bool enable_continuous();
	bool set_encodings();
	void move_buffer(yuri::ssize_t offset);

	size_t read_data_at_least(uint8_t* data, size_t size, size_t at_least);
	std::string address;
	uint16_t port;
	core::socket::pStreamSocket socket_;

	uvector<uint8_t> buffer, image;
	uint8_t *buffer_pos, *buffer_end;
	yuri::size_t buffer_size, buffer_free, buffer_valid;

	resolution_t resolution_;
	pixel_format_t pixel_format;
	receiving_states_t state;
	yuri::size_t remaining_rectangles;
	timestamp_t last_read;
	std::string socket_impl_;
};

}

}

#endif /* VNCCLIENT_H_ */
