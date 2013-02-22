/*!
 * @file 		Temperature.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef Temperature_H_
#define Temperature_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace temperature {

class Temperature: public yuri::core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Temperature();
private:
	Temperature(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* Temperature_H_ */
