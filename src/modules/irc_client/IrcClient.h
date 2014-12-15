/*!
 * @file 		IrcClient.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef IRCCLIENT_H_
#define IRCCLIENT_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/socket/StreamSocket.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
namespace yuri {
namespace irc_client {

enum class state_t {
	invalid,
	not_connected,
	verifying_nick,
	connected,
	joining_room
};

struct irc_message {
	std::string src;
	int code = -1;
	std::string cmd;
	std::string params;
};

class IrcClient: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								IrcClient(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~IrcClient() noexcept;
private:

	virtual void 				run() override;
	virtual bool 				set_param(const core::Parameter& param) override;
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	void 						process_buffer();
	void 						send_message(const std::string& msg);
	irc_message					parse_response(const std::string& resp);
	void						process_incomming_message(const std::string& msg);
	core::socket::pStreamSocket	socket_;
	std::string					socket_impl_;
	std::string					server_;
	uint16_t					port_;
	std::string					nickname_;
	std::string					alt_nickname_;
	std::string					channel_;
	std::string					target_;
	std::string					msg_buffer_;
	state_t						state_;
};

} /* namespace irc_client */
} /* namespace yuri */
#endif /* IRCCLIENT_H_ */
