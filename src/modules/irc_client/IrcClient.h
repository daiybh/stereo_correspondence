/*!
 * @file 		IrcClient.h
 * @author 		<Your name>
 * @date 		21.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef IRCCLIENT_H_
#define IRCCLIENT_H_

#include "yuri/core/BasicIOThread.h"
#include "yuri/asio/ASIOTCPSocket.h"
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

class IrcClient: public core::BasicIOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters 	configure();
	virtual 					~IrcClient();
private:
								IrcClient(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual void 				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);
	void 						process_buffer();
	void 						send_message(const std::string& msg);
	irc_message					parse_response(const std::string& resp);
	void						process_incomming_message(const std::string& msg);
	std::unique_ptr<asio::ASIOTCPSocket>
								socket_;
	std::string					server_;
	ushort_t					port_;
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
