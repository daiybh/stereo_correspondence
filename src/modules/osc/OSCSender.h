/*!
 * @file 		OSCSender.h
 * @author 		<Your name>
 * @date 		21.05.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef OSCSENDER_H_
#define OSCSENDER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/socket/DatagramSocket.h"
namespace yuri {
namespace osc {

class OSCSender: public core::IOThread, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	OSCSender(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~OSCSender() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	std::shared_ptr<core::socket::DatagramSocket> socket_;

	uint16_t	port_;
	std::string socket_type_;
	std::string address_;
};

} /* namespace osc_sender */
} /* namespace yuri */
#endif /* OSCSENDER_H_ */
