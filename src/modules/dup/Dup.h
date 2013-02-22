/*!
 * @file 		Dup.h
 * @author 		Zdenek Travnicek
 * @date 		23.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef DUP_H_
#define DUP_H_

#include "yuri/core/BasicIOThread.h"
namespace yuri {

namespace io {

class Dup: public core::BasicIOThread {
public:
	virtual ~Dup();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

	virtual void connect_out(int index,core::pBasicPipe pipe);
	virtual bool step();
	virtual bool set_param(const core::Parameter &parameter);
protected:
	Dup(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	bool hard_dup;
};

}

}

#endif /* DUP_H_ */
