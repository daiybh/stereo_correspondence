/*!
 * @file 		Overlay.h
 * @author 		<Your name>
 * @date 		27.05.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef OVERLAY_H_
#define OVERLAY_H_

#include "yuri/core/BasicIOFilter.h"

namespace yuri {
namespace overlay {

class Overlay: public core::BasicMultiIOFilter
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Overlay();
	template<class kernel>
	core::pBasicFrame combine(const core::pBasicFrame& frame_0, const core::pBasicFrame& frame_1);
private:
	Overlay(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	//virtual bool step();
	virtual std::vector<core::pBasicFrame> do_single_step(const std::vector<core::pBasicFrame>&);
	virtual bool set_param(const core::Parameter& param);
//	core::pBasicFrame frame_0;
//	core::pBasicFrame frame_1;
	ssize_t x_;
	ssize_t y_;
};

} /* namespace overlay */
} /* namespace yuri */
#endif /* OVERLAY_H_ */
