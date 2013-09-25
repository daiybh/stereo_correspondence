/*!
 * @file 		TestCard.h
 * @author 		<Your name>
 * @date 		25.09.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
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
	virtual ~TestCard();
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
