/*!
 * @file 		AudioVisualization.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef AUDIOVISUALIZATION_H_
#define AUDIOVISUALIZATION_H_

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/thread/Convert.h"

namespace yuri {
namespace audio_visualization {

class AudioVisualization: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawAudioFrame>
{
	using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawAudioFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	AudioVisualization(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~AudioVisualization() noexcept;
private:
	virtual void run() override;

	virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawAudioFrame>frames) override;
	virtual bool set_param(const core::Parameter& param) override;

	dimension_t height_;
	size_t zoom_;

	core::pConvert video_converter_;
};

} /* namespace audio_visualization */
} /* namespace yuri */
#endif /* AUDIOVISUALIZATION_H_ */
