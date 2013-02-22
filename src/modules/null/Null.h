/*!
 * @file 		Null.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef NULL_H_
#define NULL_H_

#include "yuri/core/BasicIOThread.h"

namespace yuri
{

namespace null
{

class Null: public core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Null();
protected:
	Null(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters) IO_THREAD_CONSTRUCTOR;
	virtual bool step();
};

}

}

#endif /*NULL_H_*/
