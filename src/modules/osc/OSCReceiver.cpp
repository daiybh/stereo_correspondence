/*!
 * @file 		OSCReceiver.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		13.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "OSCReceiver.h"
#include "OSC.h"
#include "yuri/core/Module.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"
#include <tuple>

namespace yuri {
namespace osc {

IOTHREAD_GENERATOR(OSCReceiver)

core::Parameters OSCReceiver::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("OSCReceiver");
	p["socket_type"]="yuri_udp";
	p["port"]=57120;
	p["address"]="0.0.0.0";
	//p->set_max_pipes(0,0);
	return p;
}


OSCReceiver::OSCReceiver(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("osc_receiver")),
event::BasicEventProducer(log),port_(2000),socket_type_("yuri_udp")
{
	set_latency(100_ms);
	IOTHREAD_INIT(parameters)


}

OSCReceiver::~OSCReceiver() noexcept
{
}


void OSCReceiver::run()
{
//	IO_THREAD_PRE_RUN
	//socket_.reset(new asio::ASIOUDPSocket(log, get_this_ptr(),port_));
	log[log::info] << "Initializing socket of type '"<< socket_type_ << "'";
	socket_ = core::DatagramSocketGenerator::get_instance().generate(socket_type_,log,"");
	log[log::info] << "Binding socket";
	if (!socket_->bind(address_,port_)) {
		log[log::fatal] << "Failed to bind socket!";
		request_end(core::yuri_exit_interrupted);
		return;
	}
	log[log::info] << "Socket initialized";
	std::vector<uint8_t> buffer(65536);
	ssize_t read_bytes=0;
	while(still_running()) {
		if (socket_->wait_for_data(get_latency())) {
			log[log::verbose_debug] << "reading data";
			read_bytes = socket_->receive_datagram(&buffer[0],buffer.size());
			if (read_bytes > 0) {
				log[log::verbose_debug] << "Read " << read_bytes << " bytes";
				auto first = buffer.begin();
//				auto events_pair = process_data(first, first + read_bytes, log);
				auto events_pair = parse_packet(first, first + read_bytes, log);
				auto events = std::get<1>(events_pair);
				if (events.empty()) continue;
				if (events.size() == 1) {
					emit_event(std::get<0>(events_pair), events[0]);
				} else {
					emit_event(std::get<0>(events_pair), make_shared<event::EventVector>(std::move(events)));
				}
			}
		}
	}
//	IO_THREAD_POST_RUN
	log[log::info] << "QUIT";
}
bool OSCReceiver::set_param(const core::Parameter& param)
{
	if (param.get_name() == "socket_type") {
		socket_type_ = param.get<std::string>();
	} else if (param.get_name() == "address") {
		address_ = param.get<std::string>();
	} else if (param.get_name() == "port") {
		port_ = param.get<uint16_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace osc_receiver */
} /* namespace yuri */
