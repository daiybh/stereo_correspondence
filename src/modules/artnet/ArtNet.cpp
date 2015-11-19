/*!
 * @file 		ArtNet.cpp
 * @author 		<Your name>
 * @date		11.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "ArtNet.h"
#include "yuri/core/Module.h"
#include "yuri/event/EventHelpers.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"
#include <boost/regex.hpp>
namespace yuri {
namespace artnet {


IOTHREAD_GENERATOR(ArtNet)

MODULE_REGISTRATION_BEGIN("artnet")
		REGISTER_IOTHREAD("artnet",ArtNet)
MODULE_REGISTRATION_END()

core::Parameters ArtNet::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Sends ArtNet packets for light control.");
	p["socket"]["Socket implementation"]="yuri_udp";
	p["address"]["Target address"]="127.0.01";
	p["port"]["Target port"]=6454;
	return p;
}


ArtNet::ArtNet(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("artnet")),
event::BasicEventConsumer(log),
socket_impl_("yuri_udp"),address_("127.0.0.1"),port_(6454),
changed_(true)
{
	IOTHREAD_INIT(parameters)
}

ArtNet::~ArtNet() noexcept
{
}

void ArtNet::run()
{
	socket_ = core::DatagramSocketGenerator::get_instance().generate(socket_impl_,log,"");
	socket_->connect(address_, port_);
	while(still_running()){
		wait_for_events(get_latency());
		process_events();
		if (changed_) {
			changed_ = false;
			for (auto& universe: universes_) {
				log[log::debug] << "Updating universe " << universe.first;
				universe.second.send(socket_);
			}
		}
	}

}

bool ArtNet::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	boost::regex universe_value("/([0-9]+)/([0-9]+)");
	boost::smatch what;
	if (boost::regex_match(event_name,what,universe_value, boost::match_default)) {
		const uint16_t universe = static_cast<uint16_t>(std::stoul(std::string{what[1].first, what[1].second}));
		const uint16_t channel = static_cast<uint16_t>(std::stoul(std::string{what[2].first, what[2].second}));
		const uint8_t value = event::lex_cast_value<uint8_t>(event);
		auto u = universes_.find(universe);
		if (u != universes_.end()) {
			u->second[channel]=value;
		} else {
			universes_[universe] = ArtNetPacket(universe);
			universes_[universe][channel]=value;
		}
//		universes_[universe][channel]=value;
		changed_ = true;
	} else if (boost::regex_match(event_name,what,universe_value_range, boost::match_default)) {
		const uint16_t universe = static_cast<uint16_t>(std::stoul(std::string{what[1].first, what[1].second}));
		const uint16_t channel_start = static_cast<uint16_t>(std::stoul(std::string{what[2].first, what[2].second}));
		const uint16_t channel_end = static_cast<uint16_t>(std::stoul(std::string{what[3].first, what[3].second}));

		std::vector<uint8_t> values;
		if (event->get_type() != event::event_type_t::vector_event) {
			values.push_back(event::lex_cast_value<uint8_t>(event));
		} else {
			const auto& vec = event::get_value<event::EventVector>(event);
			for (const auto& v: vec) {
				values.push_back(event::lex_cast_value<uint8_t>(v));
			}
			log[log::verbose_debug] << "Routing " << values.size() << " values to channels " << channel_start << " to " << channel_end;
		}
		auto u = universes_.find(universe);
		if (u == universes_.end()) {
			universes_[universe] = ArtNetPacket(universe);
		}
		auto& target_universe = universes_[universe];
		auto it = values.cbegin();
		for (auto i = channel_start; i < channel_end + 1; ++i) {
			target_universe[i]=*it++;
			if (it >= values.cend()) it = values.cbegin();
		}

		changed_ = true;
	}
	return false;
}

bool ArtNet::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(socket_impl_, "socket")
			(address_, "address")
			(port_, "port"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace artnet */
} /* namespace yuri */
