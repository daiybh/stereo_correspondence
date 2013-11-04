/*!
 * @file 		OpenCVFaceDetect.h
 * @author 		<Your name>
 * @date 		04.11.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef OPENCVFACEDETECT_H_
#define OPENCVFACEDETECT_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "yuri/event/BasicEventProducer.h"
namespace yuri {
namespace opencv {

class OpenCVFaceDetect: public core::SpecializedIOFilter<core::RawVideoFrame>, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	OpenCVFaceDetect(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~OpenCVFaceDetect() noexcept;
private:
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);

	std::string haar_cascade_file_;
	cv::CascadeClassifier haar_cascade_;
};

} /* namespace opencv_facedetection */
} /* namespace yuri */
#endif /* OPENCVFACEDETECT_H_ */
