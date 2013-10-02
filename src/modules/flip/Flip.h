/*!
 * @file 		Flip.h
 * @author 		Zdenek Travnicek
 * @date 		16.3.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef FLIP_H_
#define FLIP_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {

namespace io {

class Flip: public core::SpecializedIOFilter<core::RawVideoFrame> {
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual bool set_param(const core::Parameter &parameter);
	Flip(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Flip() noexcept;
private:
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	bool flip_x_, flip_y_;
};

}

}

#endif /* FLIP_H_ */
