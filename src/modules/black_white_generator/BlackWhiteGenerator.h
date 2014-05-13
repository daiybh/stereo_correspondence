/*!
 * @file 		BlackWhiteGenerator.h
 * @author 		<Your name>
 * @date 		13.05.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef BLACKWHITEGENERATOR_H_
#define BLACKWHITEGENERATOR_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace black_white_generator {

class BlackWhiteGenerator: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	BlackWhiteGenerator(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~BlackWhiteGenerator() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	core::pFrame black_frame_;
	core::pFrame white_frame_;
	timestamp_t start_time_;
	duration_t duration_;
	format_t format_;
	resolution_t resolution_;
};

} /* namespace black_white_generator */
} /* namespace yuri */
#endif /* BLACKWHITEGENERATOR_H_ */
