/*!
 * @file 		DXTCompress.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DXTCompress_H_
#define DXTCompress_H_

#include "yuri/io/BasicIOThread.h"

namespace yuri {
namespace dummy_module {
using yuri::log::Log;
using yuri::config::Parameter;
using yuri::config::Parameters;
using yuri::io::pThreadBase;

class DXTCompress: public yuri::io::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	virtual ~DXTCompress();
private:
	DXTCompress(Log &log_,pThreadBase parent,Parameters &parameters);
	virtual bool step();
	virtual bool set_param(Parameter& param);
	int dxt_type;
	yuri::format_t format;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DXTCompress_H_ */
