/*!
 * @file 		DXTCompress.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DXTCompress_H_
#define DXTCompress_H_

#include "yuri/core/IOThread.h"

namespace yuri {
namespace dxt_compress {


class DXTCompress: public yuri::core::IOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~DXTCompress();
private:
	DXTCompress(log::Log &log_,core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	int dxt_type;
	yuri::format_t format;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DXTCompress_H_ */
