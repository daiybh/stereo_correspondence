/*!
 * @file 		Flip.h
 * @author 		Zdenek Travnicek
 * @date 		16.3.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FLIP_H_
#define FLIP_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"
namespace yuri {

namespace io {

class Flip: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual bool set_param(const core::Parameter &parameter);
	Flip(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Flip() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	bool flip_x_, flip_y_;
};

}

}

#endif /* FLIP_H_ */
