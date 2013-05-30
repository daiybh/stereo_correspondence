/*!
 * @file 		${class_name}.cpp
 * @author 		<Your name>
 * @date		${date}
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "${class_name}.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace ${namespace} {

REGISTER("${module}",${class_name})

IO_THREAD_GENERATOR(${class_name})

core::pParameters ${class_name}::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("${class_name}");
	p->set_max_pipes(1,1);
	return p;
}


${class_name}::${class_name}(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("${module}"))
{
	IO_THREAD_INIT("${module}")
}

${class_name}::~${class_name}()
{
}

bool ${class_name}::step()
{
	core::pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		push_raw_video_frame(0, frame);
	}
	return true;
}
bool ${class_name}::set_param(const core::Parameter& param)
{
	return core::BasicIOThread::set_param(param);
}

} /* namespace ${namespace} */
} /* namespace yuri */
