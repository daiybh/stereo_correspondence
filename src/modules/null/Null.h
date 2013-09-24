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

#include "yuri/core/thread/IOFilter.h"

namespace yuri
{

namespace null
{

class Null: public core::IOFilter
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual ~Null() noexcept;
	Null(log::Log &_log,core::pwThreadBase parent, const core::Parameters& parameters);
private:
	virtual core::pFrame do_simple_single_step(const core::pFrame& frame) override;
};

}

}

#endif /*NULL_H_*/
