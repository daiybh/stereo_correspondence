/*!
 * @file 		SineGenerator.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SINEGENERATOR_H_
#define SINEGENERATOR_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace audio_gen {

class SineGenerator: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	SineGenerator(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SineGenerator() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	double frequency_;
	duration_t t_;
	size_t sample_;
	size_t channels_;
	size_t sampling_;
	size_t samples_;
};

} /* namespace audio_gen */
} /* namespace yuri */
#endif /* SINEGENERATOR_H_ */
