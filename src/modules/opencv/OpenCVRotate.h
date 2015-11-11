/*!
 * @file 		OpenCV.h
 * @author 		Jiri Melnikov
 * @date 		15.5.2015
 * @date		16.5.2015
 * @copyright	CESNET, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_OPENCV_OPENCVROTATE_H_
#define SRC_MODULES_OPENCV_OPENCVROTATE_H_

#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/utils/color.h"
#include "opencv2/imgproc/imgproc.hpp"
namespace yuri {
namespace opencv {

class OpenCVRotate: public core::SpecializedIOFilter<core::RawVideoFrame>,
public event::BasicEventConsumer
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	virtual ~OpenCVRotate() noexcept;
	OpenCVRotate(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	double angle_;
	core::color_t color_;
};

} /* namespace dummy_module */
} /* namespace yuri */
#endif /* OpenCV_H_ */
