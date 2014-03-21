/*!
 * @file 		AudioNoise.h
 * @author 		<Your name>
 * @date 		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef AUDIONOISE_H_
#define AUDIONOISE_H_

#include "yuri/core/thread/IOThread.h"
#include <random>
namespace yuri {
namespace audio_noise {

class AudioNoise: public core::IOThread
{
public:
	struct frame_generator_t;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	AudioNoise(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~AudioNoise() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	format_t format_;
	double amplitude_;
	std::random_device rand_;
	std::unique_ptr<frame_generator_t> generator_;
	size_t sampling_frequency_;
	size_t channels_;
};

} /* namespace audio_noise */
} /* namespace yuri */
#endif /* AUDIONOISE_H_ */
