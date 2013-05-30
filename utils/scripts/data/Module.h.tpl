/*!
 * @file 		${class_name}.h
 * @author 		<Your name>
 * @date 		${date}
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef ${guard}_H_
#define ${guard}_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace ${namespace} {

class ${class_name}: public core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~${class_name}();
private:
	${class_name}(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace ${namespace} */
} /* namespace yuri */
#endif /* ${guard}_H_ */
