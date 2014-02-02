/*!
 * @file 		UVAudioTestcard.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVAUDIOTESTCARD_H_
#define UVAUDIOTESTCARD_H_

#include "yuri/core/thread/IOThread.h"

namespace yuri {
namespace uv_audio_testcard {

class UVAudioTestcard: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	UVAudioTestcard(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~UVAudioTestcard() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	void* device_;
	size_t capture_channels_;
};

} /* namespace uv_audio_testcard */
} /* namespace yuri */
#endif /* UVAUDIOTESTCARD_H_ */
