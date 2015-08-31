/*!
 * @file 		Mosaic.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef MOSAIC_H_
#define MOSAIC_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
namespace mosaic {

struct mosaic_detail_t {
	position_t radius;
	position_t tile_size;
	coordinates_t center;
};

class Mosaic: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Mosaic(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Mosaic() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	std::vector<mosaic_detail_t> mosaics_;
};

} /* namespace mosaic */
} /* namespace yuri */
#endif /* MOSAIC_H_ */
