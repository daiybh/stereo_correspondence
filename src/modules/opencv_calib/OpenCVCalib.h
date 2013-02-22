/*!
 * @file 		OpenCVCalib.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef OpenCVCalib_H_
#define OpenCVCalib_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace opencv {

class OpenCVCalib: public yuri::core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~OpenCVCalib();
private:
	OpenCVCalib(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	format_t format;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* OpenCVCalib_H_ */
