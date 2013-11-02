/*!
 * @file 		Mosaic.h
 * @author 		<Your name>
 * @date 		02.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef MOSAIC_H_
#define MOSAIC_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/Convert.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace mosaic {

class Mosaic: public core::SpecializedIOFilter<core::VideoFrame>, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Mosaic(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Mosaic() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	core::pConvert converter_;
	position_t radius_;
	position_t tile_size_;
	coordinates_t center_;
};

} /* namespace mosaic */
} /* namespace yuri */
#endif /* MOSAIC_H_ */
