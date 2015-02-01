/*!
 * @file 		TheoraEncoder.h
 * @author 		<Your name>
 * @date 		21.03.2014
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef THEORAENCODER_H_
#define THEORAENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <theora/theoraenc.h>

namespace yuri {
namespace theora {

class TheoraEncoder: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
	using ctx_handle_t = std::unique_ptr<th_enc_ctx, std::function<void(th_enc_ctx*)>>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	TheoraEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~TheoraEncoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	bool init_ctx(const core::pRawVideoFrame& frame);
	void process_packet(ogg_packet& packet);
	th_info theora_info_;
	ctx_handle_t ctx_;
	ogg_stream_state ogg_state_;
	bool mux_into_ogg_;
	bool low_latency_;
	int quality_;
	int bitrate_;
	int stream_id_;
	int fast_encoding_;
	duration_t frame_duration_;
};

} /* namespace theora */
} /* namespace yuri */
#endif /* THEORAENCODER_H_ */
