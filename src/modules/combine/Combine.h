/*!
 * @file 		Combine.h
 * @author 		Zdenek Travnicek
 * @date		7.3.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef COMBINE_H_
#define COMBINE_H_

#include "yuri/core/BasicIOThread.h"
#include <vector>
namespace yuri {
namespace combine {

class Combine: public core::BasicIOThread
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	virtual ~Combine();
private:
	Combine(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	size_t x,y;
	std::vector<core::pBasicFrame> frames;
};

} /* namespace combine */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
