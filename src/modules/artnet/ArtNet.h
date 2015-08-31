/*!
 * @file 		ArtNet.h
 * @author 		<Your name>
 * @date 		11.12.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef ARTNET_H_
#define ARTNET_H_

#include "ArtNetPacket.h"
#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include <unordered_map>
namespace yuri {
namespace artnet {

class ArtNet: public core::IOThread, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	ArtNet(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~ArtNet() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	std::string socket_impl_;
	core::socket::pDatagramSocket socket_;
	std::string address_;
	core::socket::port_t port_;
	bool changed_;

	std::unordered_map<uint16_t, ArtNetPacket> universes_;


};

} /* namespace artnet */
} /* namespace yuri */
#endif /* ARTNET_H_ */
