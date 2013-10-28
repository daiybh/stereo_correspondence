/*!
 * @file 		FrameInfo.h
 * @author 		<Your name>
 * @date 		28.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef FRAMEINFO_H_
#define FRAMEINFO_H_

#include "yuri/core/thread/IOFilter.h"

namespace yuri {
namespace frame_info {

class FrameInfo: public core::IOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	FrameInfo(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FrameInfo() noexcept;
private:
	
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame);
	virtual bool set_param(const core::Parameter& param);
	core::pFrame last_frame_;
	bool print_all_;
};

} /* namespace frame_info */
} /* namespace yuri */
#endif /* FRAMEINFO_H_ */
