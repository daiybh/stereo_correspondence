/*!
 * @file 		Split.h
 * @author 		Zdenek Travnicek
 * @date 		30.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef SPLIT_H_
#define SPLIT_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri {

namespace split {

class Split: public core::BasicIOThread
{
public:
	Split(log::Log &_log, core::pwThreadBase parent,core::Parameters &params);
	virtual ~Split();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
protected:
	virtual bool step();
};

}

}

#endif /* SPLIT_H_ */
