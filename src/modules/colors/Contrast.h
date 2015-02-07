/*!
 * @file 		Contrast.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef CONTRAST_H_
#define CONTRAST_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"


namespace yuri {
namespace colors {

class Contrast: public core::SpecializedIOFilter<core::RawVideoFrame>,
public event::BasicEventConsumer
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Contrast(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Contrast() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;


	double contrast_;
	bool crop_;
};

} /* namespace contrast */
} /* namespace yuri */
#endif /* CONTRAST_H_ */
