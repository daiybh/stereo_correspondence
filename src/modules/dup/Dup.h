/*!
 * @file 		Dup.h
 * @author 		Zdenek Travnicek
 * @date 		23.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DUP_H_
#define DUP_H_

#include "yuri/core/thread/MultiIOFilter.h"
namespace yuri {

namespace io {

class Dup: public core::MultiIOFilter {
public:
	Dup(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Dup() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual bool set_param(const core::Parameter &parameter);

private:
	virtual std::vector<core::pFrame> do_single_step(std::vector<core::pFrame> frames) override;
	virtual void do_connect_out(position_t index, core::pPipe pipe) override;
	bool hard_dup_;
};

}

}

#endif /* DUP_H_ */
