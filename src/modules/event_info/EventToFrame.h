/*!
 * @file 		EventToFrame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		12.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENTTOFRAME_H_
#define EVENTTOFRAME_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"

namespace yuri {
namespace event_to_frame {

class EventToFrame: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	EventToFrame(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~EventToFrame() noexcept;
private:
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	virtual void receive_event_hook() noexcept override;
};

} /* namespace event_to_frame */
} /* namespace yuri */
#endif /* EVENTTOFRAME_H_ */
