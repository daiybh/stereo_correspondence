/*!
 * @file 		DummyModule.h
 * @author 		Zdenek Travnicek
 * @date		17.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DUMMYMODULE_H_
#define DUMMYMODULE_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace imagemagick_module {

class ImageMagickSource: public core::SpecializedIOFilter<core::CompressedVideoFrame>, public core::ConverterThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual ~ImageMagickSource() noexcept;
	ImageMagickSource(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters);
private:
	virtual core::pFrame do_convert_frame(core::pFrame input_frame, format_t target_format) override;
	virtual core::pFrame do_special_single_step(core::pCompressedVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param)  override;
	yuri::format_t format_;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* DUMMYMODULE_H_ */
