/*!
 * @file 		Diff.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef _H_
#define _H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {
namespace diff {

class Diff: public core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Diff();
private:
	Diff(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters);
	virtual bool step();
	core::pBasicFrame frame1;
	core::pBasicFrame frame2;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* _H_ */
