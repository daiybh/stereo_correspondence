/*!
 * @file 		Magnify.h
 * @author 		<Your name>
 * @date 		31.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef MAGNIFY_H_
#define MAGNIFY_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"
namespace yuri {
namespace magnify {

class Magnify: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventConsumer
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Magnify(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Magnify() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	geometry_t geometry_;
	size_t	zoom_;
};

} /* namespace magnify */
} /* namespace yuri */
#endif /* MAGNIFY_H_ */
