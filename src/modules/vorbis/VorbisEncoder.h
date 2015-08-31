/*!
 * @file 		VorbisEncoder.h
 * @author 		<Your name>
 * @date 		25.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef VORBISENCODER_H_
#define VORBISENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include <vorbis/vorbisenc.h>
namespace yuri {
namespace vorbis {

enum class encoding_type_t {
	vbr,
	cbr,
	abr
};

class VorbisEncoder: public core::SpecializedIOFilter<core::RawAudioFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawAudioFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	VorbisEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~VorbisEncoder() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawAudioFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	bool initialize(const core::pRawAudioFrame& frame);
	bool process_packet(ogg_packet& packet);
	vorbis_info info_;
	vorbis_dsp_state state_;
	vorbis_block block_;
	ogg_stream_state ogg_state_;
	bool initialized_;
	int bitrate_;
	float quality_;
	encoding_type_t encoding_type_;
	int encoding_channels_;
};

} /* namespace vorbis */
} /* namespace yuri */
#endif /* VORBISENCODER_H_ */
