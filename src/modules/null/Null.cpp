/*!
 * @file 		Null.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Null.h"
#include "yuri/core/Module.h"
namespace yuri
{

namespace null
{

IOTHREAD_GENERATOR(Null)

MODULE_REGISTRATION_BEGIN("null")
		REGISTER_IOTHREAD("null",Null)
MODULE_REGISTRATION_END()

core::Parameters Null::configure()
{
	core::Parameters p = IOFilter::configure();
	return p;
}

Null::Null(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters)
		:core::IOFilter(log_,parent,"Null")
{
	IOTHREAD_INIT(parameters)
	set_latency(100_ms);
	resize(1,0);
}

Null::~Null() noexcept
{
}

core::pFrame Null::do_simple_single_step(const core::pFrame&)
{
	return {};
}

}

}
