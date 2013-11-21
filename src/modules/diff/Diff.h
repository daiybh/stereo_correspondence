/*!
 * @file 		Diff.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef _H_
#define _H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace diff {

class Diff: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Diff(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Diff() noexcept;
private:

	virtual bool step();
	core::pRawVideoFrame frame1;
	core::pRawVideoFrame frame2;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* _H_ */
