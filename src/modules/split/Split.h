/*!
 * @file 		Split.h
 * @author 		Zdenek Travnicek
 * @date 		30.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2011 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SPLIT_H_
#define SPLIT_H_

#include "yuri/core/IOThread.h"
#include "yuri/core/BasicIOFilter.h"
namespace yuri {

namespace split {

class Split: public core::BasicMultiIOFilter
{
public:
	Split(log::Log &_log, core::pwThreadBase parent,core::Parameters &params);
	virtual ~Split();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
private:
	virtual std::vector<core::pBasicFrame> do_single_step(const std::vector<core::pBasicFrame>& frames);
	virtual bool 			set_param(const core::Parameter &parameter);
	size_t	x_;
	size_t	y_;
};

}

}

#endif /* SPLIT_H_ */
