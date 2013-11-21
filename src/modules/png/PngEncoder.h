/*!
 * @file 		PngEncoder.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PNGENCODER_H_
#define PNGENCODER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace png {

class PngEncoder: public core::SpecializedIOFilter<core::RawVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	PngEncoder(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~PngEncoder() noexcept;
private:
	
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual bool set_param(const core::Parameter& param);
};

} /* namespace png */
} /* namespace yuri */
#endif /* PNGENCODER_H_ */
