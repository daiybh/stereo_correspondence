/*!
 * @file 		FrameInfo.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		28.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FRAMEINFO_H_
#define FRAMEINFO_H_

#include "yuri/core/thread/IOFilter.h"
#include "yuri/event/BasicEventProducer.h"

namespace yuri {
namespace frame_info {

class FrameInfo: public core::IOFilter, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	FrameInfo(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FrameInfo() noexcept;
private:
	
	virtual core::pFrame do_simple_single_step(core::pFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	core::pFrame last_frame_;
	bool print_all_;
	bool print_time_;

	size_t frame_count_;
	std::string info_string_;
	std::string print_string_;
};

} /* namespace frame_info */
} /* namespace yuri */
#endif /* FRAMEINFO_H_ */
