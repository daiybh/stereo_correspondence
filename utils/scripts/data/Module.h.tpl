/*!
 * @file 		${class_name}.h
 * @author 		<Your name>
 * @date 		${date}
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef ${guard}_H_
#define ${guard}_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace ${namespace} {

class ${class_name}: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	${class_name}(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~${class_name}();
private:
	
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace ${namespace} */
} /* namespace yuri */
#endif /* ${guard}_H_ */
