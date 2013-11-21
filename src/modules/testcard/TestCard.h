/*!
 * @file 		TestCard.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		25.09.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef TESTCARD_H_
#define TESTCARD_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace testcard {

class TestCard: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	TestCard(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~TestCard() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	resolution_t	resolution_;
	double			fps_;
	format_t		format_;
};

} /* namespace testcard */
} /* namespace yuri */
#endif /* TESTCARD_H_ */
