/*
 * OpenCVSource.h
 *
 *  Created on: 1. 2. 2015
 *      Author: neneko
 */

#ifndef SRC_MODULES_OPENCV_OPENCVSOURCE_H_
#define SRC_MODULES_OPENCV_OPENCVSOURCE_H_

#include "yuri/core/thread/IOThread.h"
#include <opencv2/highgui/highgui.hpp>
namespace yuri {
namespace opencv {

class OpenCVSource: public core::IOThread
{
public:
	static core::Parameters configure();
	IOTHREAD_GENERATOR_DECLARATION
	OpenCVSource(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters);
	~OpenCVSource() noexcept;

private:
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

	int device_index_;
	std::string device_path_;

	cv::VideoCapture capture_;
        int width,height;
};


}
}


#endif /* SRC_MODULES_OPENCV_OPENCVSOURCE_H_ */
