/*!
 * @file 		${class_name}.h
 * @author 		${user}
 * @date 		${date}
 * @copyright	Institute of Intermedia, CTU in Prague, ${year}
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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
	${class_name}(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~${class_name}() noexcept;
private:
	
	virtual bool step() override;
	virtual bool set_param(const core::Parameter& param) override;
};

} /* namespace ${namespace} */
} /* namespace yuri */
#endif /* ${guard}_H_ */
