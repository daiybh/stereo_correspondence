/*!
 * @file 		SpecialPipes.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SPECIALPIPES_H_
#define SPECIALPIPES_H_

#include "yuri/core/pipe/PipeGenerator.h"
#include "yuri/core/pipe/PipePolicies.h"
#include <deque>
namespace yuri {
namespace core {

template <template <bool> class Policy, bool blocking>
class SpecialPipe: public Pipe, public Policy<blocking> {
public:
								SpecialPipe(const std::string& name, const log::Log& log_, const Parameters& params)
					:Pipe(name, log_),Policy<blocking>(params) {}
								~SpecialPipe() noexcept {}
	static pPipe 				generate(const std::string& name, const log::Log& log_, const Parameters& params) {
		return std::make_shared<SpecialPipe<Policy, blocking>>(name, log_, params);
	}
private:
	virtual bool 				do_push_frame(const pFrame &frame) override
	{
		return Policy<blocking>::impl_push_frame(frame);
	}
	virtual pFrame 				do_pop_frame() override
	{
		return Policy<blocking>::impl_pop_frame();
	}
	virtual size_t				do_get_size() const override
	{
		return Policy<blocking>::impl_get_size();
	}
	virtual void 				drop_frame(const pFrame& frame) override
	{
		Pipe::drop_frame(frame);
	}
	virtual bool				do_is_blocking() const noexcept override
	{
		return blocking;
	}
	virtual bool 				do_is_full() const noexcept override
	{
		return Policy<blocking>::impl_is_full();
	}
};

using BlockingUnlimitedPipe 		= SpecialPipe<pipe::UnlimitedPolicy, 	true>;
using BlockingSingleFramePipe 		= SpecialPipe<pipe::SingleFramePolicy, 	true>;
using BlockingSizeLimitedPipe 		= SpecialPipe<pipe::SizeLimitedPolicy, 	true>;
using BlockingCountLimitedPipe 		= SpecialPipe<pipe::CountLimitedPolicy, true>;
using NonBlockingUnlimitedPipe 		= SpecialPipe<pipe::UnlimitedPolicy, 	false>;
using NonBlockingSingleFramePipe 	= SpecialPipe<pipe::SingleFramePolicy, 	false>;
using NonBlockingSizeLimitedPipe 	= SpecialPipe<pipe::SizeLimitedPolicy, 	false>;
using NonBlockingCountLimitedPipe 	= SpecialPipe<pipe::CountLimitedPolicy, false>;

}
}


#endif /* SPECIALPIPES_H_ */
