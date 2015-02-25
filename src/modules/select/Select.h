/*!
 * @file 		Select.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.03.2014
 * @copyright	Institute of Intermedia, 2014
 * 				Distributed BSD License
 *
 */

#ifndef SELECT_H_
#define SELECT_H_

#include "yuri/core/thread/MultiIOFilter.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace select {

class Select: public core::IOThread, public event::BasicEventConsumer
{
	using base_type = core::IOThread;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Select(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Select() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual void do_connect_in(position_t, core::pPipe pipe) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	position_t index_;
};

} /* namespace select */
} /* namespace yuri */
#endif /* SELECT_H_ */
