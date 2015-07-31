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

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {

namespace split {

class Split: public core::SpecializedMultiIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame>;
public:
	Split(log::Log &_log, core::pwThreadBase parent, core::Parameters params);
	virtual ~Split() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
private:
	virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame> frames) override;
	virtual bool 			set_param(const core::Parameter &parameter) override;
	size_t	x_;
	size_t	y_;
};

}

}

#endif /* SPLIT_H_ */
