/*!
 * @file 		Unselect.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		14.12.2014
 * @copyright	Institute of Intermedia, 2014
 * 				Distributed BSD License
 *
 */

#ifndef SRC_MODULES_Unselect_UNUnselect_H_
#define SRC_MODULES_Unselect_UNUnselect_H_

#include "yuri/core/thread/MultiIOFilter.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace select {

class Unselect: public core::IOThread, public event::BasicEventConsumer
{
	using base_type = core::IOThread;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Unselect(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Unselect() noexcept;
private:

	virtual bool step() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual void do_connect_out(position_t, core::pPipe pipe) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	position_t index_;
};

} /* namespace Unselect */
} /* namespace yuri */


#endif /* SRC_MODULES_Unselect_UNUnselect_H_ */
