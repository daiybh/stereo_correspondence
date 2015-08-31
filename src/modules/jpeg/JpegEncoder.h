/*!
 * @file 		JpegEncoder.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		31.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef JPEGENCODER_H_
#define JPEGENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
#include "yuri/core/thread/Convert.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/event/BasicEventConsumer.h"
namespace yuri {
namespace jpeg {

class JpegEncoder: public core::SpecializedIOFilter<core::RawVideoFrame>,
public core::ConverterThread,
public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	JpegEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~JpegEncoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	size_t quality_;
	bool force_mjpeg_;
};

} /* namespace jpeg */
} /* namespace yuri */
#endif /* JPEGENCODER_H_ */
