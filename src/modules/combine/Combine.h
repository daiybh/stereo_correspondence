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

#include "yuri/core/thread/MultiIOFilter.h"
#include <vector>
namespace yuri {
namespace combine {

class Combine: public core::MultiIOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Combine(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Combine() noexcept;
private:
	virtual std::vector<core::pFrame> do_single_step(const std::vector<core::pFrame>& frames) override;
	virtual bool set_param(const core::Parameter& param);
	size_t x_,y_;

};

} /* namespace combine */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
