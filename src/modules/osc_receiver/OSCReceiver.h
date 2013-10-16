/*!
 * @file 		OSCReceiver.h
 * @author 		<Your name>
 * @date 		13.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef OSCRECEIVER_H_
#define OSCRECEIVER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventProducer.h"
#include "yuri/core/socket/DatagramSocket.h"

namespace yuri {
namespace osc_receiver {

class OSCReceiver: public core::IOThread, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	OSCReceiver(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~OSCReceiver() noexcept;
private:

	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	template<class Iterator>
	void process_data(Iterator& first, const Iterator& last);
	std::shared_ptr<core::socket::DatagramSocket> socket_;
	uint16_t	port_;
	std::string socket_type_;
};

} /* namespace osc_receiver */
} /* namespace yuri */
#endif /* OSCRECEIVER_H_ */
