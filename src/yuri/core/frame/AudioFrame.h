/*!
 * @file 		AudioFrame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.10.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef AUDIOFRAME_H_
#define AUDIOFRAME_H_
#include "Frame.h"

namespace yuri {
namespace core {

class AudioFrame;
typedef shared_ptr<AudioFrame> pAudioFrame;

EXPORT class AudioFrame: public Frame
{
public:
	AudioFrame(format_t format, size_t channel_count, size_t sampling_frequency);
	~AudioFrame() noexcept;

	size_t						get_channel_count() const { return channel_count_; }
	void						set_channel_count(size_t channel_count);

	size_t						get_sampling_frequency() const { return sampling_frequency_; }
	void						set_sampling_frequency(size_t sampling_frequency_);

private:
	size_t 						channel_count_;
	size_t						sampling_frequency_;

};

}
}



#endif /* AUDIOFRAME_H_ */
