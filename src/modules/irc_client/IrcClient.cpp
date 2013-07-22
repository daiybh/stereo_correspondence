/*!
 * @file 		IrcClient.cpp
 * @author 		<Your name>
 * @date		21.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "IrcClient.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"
#include <poll.h>

namespace yuri {
namespace irc_client {

REGISTER("irc_client",IrcClient)

IO_THREAD_GENERATOR(IrcClient)

core::pParameters IrcClient::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("IrcClient");
	p->set_max_pipes(0,0);
	(*p)["server"]["Server hostname or IP address"]=std::string();
	(*p)["nickname"]["Nickname to use at server"]=std::string("yuri_irc");
	(*p)["alt_nickname"]["Alternative nickname to use at server (optional)"]=std::string();
	(*p)["port"]["Server port"]=6667;
	(*p)["channel"]["Channel name to join (optional)"]=std::string();
	(*p)["target"]["Nickname or channel name, where the messages should be sent (optional)"]=std::string();
	return p;
}


IrcClient::IrcClient(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,0,0,std::string("irc_client")),
event::BasicEventConsumer(log),
event::BasicEventProducer(log),
port_(6667),state_(state_t::not_connected)
{
	IO_THREAD_INIT("irc_client")
}

IrcClient::~IrcClient()
{
}

void IrcClient::run()
{
	IO_THREAD_PRE_RUN
	socket_.reset(new asio::ASIOTCPSocket(log,get_this_ptr()));
	if (!socket_->connect(server_,port_)) {
		log[log::error] << "Failed to connect to " << server_ << ":" << port_;
		request_end();
	}
	send_message("NICK " + nickname_);
	send_message("USER " + nickname_ + " " + nickname_ + " " + server_ + ": YURI");
	state_ = state_t::verifying_nick;
	std::vector<ubyte_t> buffer(4096,0);
	size_t read_bytes;
	pollfd fds = {socket_->get_fd(), POLLIN, 0};

	while(still_running()) {
		process_events();
		poll(&fds,1,10);
		if (fds.revents==POLLIN)
		/*if (socket_->data_available() == 0) {
//			log[log::info] << "No data ";
			sleep(10);
		} else */{
//			log[log::info] << "Reading";
			read_bytes = socket_->read(&buffer[0],buffer.size());
//			log[log::info] << "Read " << read_bytes;
			if (read_bytes > 0) {
//				log[log::info] << "Read " << read_bytes << " bytes";
				msg_buffer_.insert(msg_buffer_.end(),buffer.begin(),buffer.begin()+read_bytes);
				process_buffer();
				if (state_==state_t::invalid) request_end();
			}
		}
	}
	if (socket_) send_message("QUIT : Good bye, cruel world");
	IO_THREAD_POST_RUN
}

void IrcClient::process_buffer()
{
//	log[log::info] << " Buffer is: " << msg_buffer_;
	auto idx = msg_buffer_.find_first_of("\n\r");
	while (idx != std::string::npos) {
		std::string msg = msg_buffer_.substr(0, idx);
//		log[log::info] << "Found line: " << msg;
		msg_buffer_=msg_buffer_.substr(idx+1);
		idx = msg_buffer_.find_first_of("\n\r");

		irc_message resp = parse_response(msg);
		switch (state_) {
			case state_t::verifying_nick:
					if (resp.code > 400) {
						log[log::error] << "Failed to set nick to " + nickname_;
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
//	log[log::info] << "parsing the rest: '" << resp << "'";
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
	std::vector<ubyte_t> data(msg.begin(),msg.end());
	data.push_back('\r');
	data.push_back('\n');
	socket_->write(&data[0],data.size());
}
void IrcClient::process_incomming_message(const std::string& msg)
{
	log[log::debug] << "Someone told me: " << msg;
	auto idx = msg.find_first_of(' ');
	if (idx==std::string::npos) return;
	std::string cmd = msg.substr(0,idx);
	if (iequals(cmd,"send")) {
		auto idx2 = msg.find_first_not_of(' ',idx);
		auto idx3 = msg.find_first_of(' ',idx2);
		if (idx2 == std::string::npos || idx3 == std::string::npos) return;
		std::string tgt = msg.substr(idx2,idx3-idx2);
		auto idx4 = msg.find_first_not_of(' ',idx3);
		emit_event(tgt,make_shared<event::EventString>(msg.substr(idx4)));
	}
}
bool IrcClient::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	log[log::info] << "Received " << event_name;
	if (state_ == state_t::connected && !target_.empty()) {
		if (event->get_type() == event::event_type_t::string_event) {
			send_message("PRIVMSG " + target_ + " :"+event_name+": "+event::get_value<event::EventString>(event));
		} else {
			send_message("PRIVMSG " + target_ + " : "+"Received event "+event_name);
		}
	}
	return true;
}
bool IrcClient::set_param(const core::Parameter& param)
{
	if (iequals(param.name,"server")) {
		server_ = param.get<std::string>();
	} else if (iequals(param.name,"nickname")) {
		nickname_ = param.get<std::string>();
	} else if (iequals(param.name,"alt_nickname")) {
		alt_nickname_ = param.get<std::string>();
	} else if (iequals(param.name,"port")) {
		port_ = param.get<ushort_t>();
	} else if (iequals(param.name,"channel")) {
		channel_ = param.get<std::string>();
	} else if (iequals(param.name,"target")) {
		target_ = param.get<std::string>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace irc_client */
} /* namespace yuri */
