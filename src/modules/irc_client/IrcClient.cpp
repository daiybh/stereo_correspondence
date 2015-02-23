/*!
 * @file 		IrcClient.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		21.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "IrcClient.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"
#include "yuri/core/socket/StreamSocketGenerator.h"
#include <cassert>

namespace yuri {
namespace irc_client {

MODULE_REGISTRATION_BEGIN("irc_client")
	REGISTER_IOTHREAD("irc_client",IrcClient)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(IrcClient)

core::Parameters IrcClient::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Connects to irc server and sends all incomming events to target (username or channel), if specified. Also send all private messages starting with 'send' back as events.");
//	p->set_max_pipes(0,0);
	p["server"]["Server hostname or IP address"]=std::string();
	p["nickname"]["Nickname to use at server"]=std::string("yuri_irc");
	p["alt_nickname"]["Alternative nickname to use at server (optional)"]=std::string();
	p["port"]["Server port"]=6667;
	p["channel"]["Channel name to join (optional)"]=std::string();
	p["target"]["Nickname or channel name, where the messages should be sent (optional)"]=std::string();
	return p;
}


IrcClient::IrcClient(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("irc_client")),
event::BasicEventConsumer(log),
event::BasicEventProducer(log),
socket_impl_("yuri_tcp"),port_(6667),state_(state_t::not_connected)
{
	IOTHREAD_INIT(parameters)
}

IrcClient::~IrcClient() noexcept
{
}

void IrcClient::run()
{
	socket_ = core::StreamSocketGenerator::get_instance().generate(socket_impl_, log);
	socket_->bind("",0);
	if (!socket_->connect(server_,port_)) {
		log[log::error] << "Failed to connect to " << server_ << ":" << port_;
		request_end();
	}
	send_message("NICK " + nickname_);
//	send_message("USER " + nickname_ + " " + nickname_ + " " + server_ + ": YURI");
	send_message("USER " + nickname_ + " 0 * " + ":YURI");
	state_ = state_t::verifying_nick;
	std::vector<uint8_t> buffer(4096,0);
	size_t read_bytes;

	while(still_running()) {
		process_events();
		if (socket_->wait_for_data(get_latency())) {
			read_bytes = socket_->receive_data(buffer);
			if (read_bytes > 0) {
				log[log::info] << "Read " << read_bytes << " bytes";
				msg_buffer_.insert(msg_buffer_.end(),buffer.begin(),buffer.begin()+read_bytes);
				process_buffer();
				if (state_==state_t::invalid) request_end();
			}
		}
	}
	if (socket_) send_message("QUIT : Good bye, cruel world");
}

void IrcClient::process_buffer()
{
	log[log::info] << " Buffer is: " << msg_buffer_;
	auto idx = msg_buffer_.find_first_of("\n\r");
	while (idx != std::string::npos) {
		std::string msg = msg_buffer_.substr(0, idx);
		msg_buffer_=msg_buffer_.substr(idx+1);
		idx = msg_buffer_.find_first_of("\n\r");

		irc_message resp = parse_response(msg);
		switch (state_) {
			case state_t::verifying_nick:
					if (resp.code > 400) {
						log[log::error] << "Failed to set nick to " + nickname_ + " ("+resp.src+")";
						state_ = state_t::invalid;
					} else if (resp.code > 0 && resp.code < 100) {
						state_ = state_t::connected;
						if (!channel_.empty()) send_message("JOIN " + channel_);
					}
					break;
			case state_t::connected:
				if (!resp.cmd.empty()) {
					if (iequals(resp.cmd,"ping")) {
						log[log::info] << "Replying to ping from " << resp.src;
						send_message("PONG " + resp.params.substr(1));
					} else if (iequals(resp.cmd,"privmsg")) {
						log[log::info] << "Received message from " << resp.src;
						auto idx1 = resp.params.find_first_not_of(' ');
						auto idx2 = resp.params.find_first_of(' ', idx1);
						log[log::info] << "Msg target: " << resp.params.substr(idx1,idx2) << ", rest is: " << resp.params.substr(idx2+1);
						if (iequals(resp.params.substr(idx1,idx2),nickname_)) {
							auto idx3 = resp.params.find_first_not_of(" :", idx2);
							process_incomming_message(resp.params.substr(idx3));
						}
					} else {
						log[log::info] << "Unknown command " << resp.cmd;
					}

				} else {
//					log[log::info] << "Unexpected message" << resp.params;
				}
				break;
			default:
				state_ = state_t::invalid;
				break;
		}

	}
	msg_buffer_.clear();

}
irc_message IrcClient::parse_response(const std::string& resp)
{
	if (resp[0] == '\r') return parse_response(resp.substr(1));
	if (resp[0] == ':') {
		auto idx = resp.find_first_of(' ');
		if (idx==std::string::npos) return irc_message();
//		log[log::info] << "Found src: " << resp.substr(1,idx);
		irc_message msg = parse_response(resp.substr(idx+1));
		msg.src = resp.substr(1,idx);
		return msg;
	}
	auto idx = resp.find_first_not_of(' ');
	if (idx==std::string::npos) return irc_message();
	if (idx>0) return parse_response(resp.substr(idx+1));
	//if (resp[0]<'0' || resp[0]>'9') return -1;
	idx = resp.find_first_of(' ');
	irc_message msg;
	try {
		msg.code = std::stoi(resp.substr(0,idx));
		return msg;
	}
	catch (const std::invalid_argument&) {}
	msg.cmd = resp.substr(0,idx);
	msg.params = resp.substr(idx+1);
	return msg;
}
void IrcClient::send_message(const std::string& msg)
{
	assert(socket_);
	log[log::info] << "MESSAGE: '"<<msg<<"'";
	std::vector<uint8_t> data(msg.begin(),msg.end());
	data.push_back('\r');
	data.push_back('\n');
	socket_->send_data(data);
}
void IrcClient::process_incomming_message(const std::string& msg)
{
	log[log::debug] << "Someone told me: " << msg;
	auto idx = msg.find_first_of(' ');
	if (idx==std::string::npos) return;
	std::string cmd = msg.substr(0,idx);
	if (iequals(cmd,"send")) {
		auto idx2 = msg.find_first_not_of(' ',idx);
		// no event for us
		if (idx2 == std::string::npos) return;
		auto idx3 = msg.find_first_of(' ',idx2);
		//if (idx2 == std::string::npos || idx3 == std::string::npos) return;
		if (idx3 == std::string::npos) {
			// no event value, so let's fire BANG
			emit_event(msg.substr(idx2));
		} else {
			const std::string tgt = msg.substr(idx2,idx3-idx2);
			auto idx4 = msg.find_first_not_of(' ',idx3);
			emit_event(tgt,msg.substr(idx4));
		}
	}
}
bool IrcClient::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	log[log::info] << "Received " << event_name;
	if (state_ == state_t::connected && !target_.empty()) {
		send_message("PRIVMSG " + target_ + " :"+event_name+": "+event::lex_cast_value<std::string>(event));
//		} else {
//			send_message("PRIVMSG " + target_ + " : "+"Received event "+event_name);
//		}
	}
	return true;
}
bool IrcClient::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(server_, 		"server")
			(nickname_, 	"nickname")
			(alt_nickname_, "alt_nickname")
			(port_, 		"port")
			(channel_, 		"channel")
			(target_,		"target")
			(socket_impl_,	"socket"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace irc_client */
} /* namespace yuri */
