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

#include "yuri/core/BasicIOThread.h"
#include "yuri/event/BasicEventProducer.h"
#include "yuri/asio/ASIOUDPSocket.h"
namespace yuri {
namespace osc_receiver {

class OSCReceiver: public core::BasicIOThread, public event::BasicEventProducer
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~OSCReceiver();
private:
	OSCReceiver(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual void run();
	virtual bool set_param(const core::Parameter& param);
	template<class Iterator>
	void process_data(Iterator& first, const Iterator& last);
	std::unique_ptr<asio::ASIOUDPSocket> socket_;
	ushort_t	port_;
};

} /* namespace osc_receiver */
} /* namespace yuri */
#endif /* OSCRECEIVER_H_ */
