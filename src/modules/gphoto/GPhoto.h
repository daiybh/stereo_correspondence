/*!
 * @file 		GPhoto.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		29.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef GPHOTO_H_
#define GPHOTO_H_

#include "yuri/core/thread/IOThread.h"
#include <gphoto2/gphoto2-camera.h>

namespace yuri {
namespace gphoto {

class GPhoto: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	GPhoto(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~GPhoto() noexcept;
private:
	bool open_camera();
	bool close_camera();
	bool enable_capture();
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;

private:
	Camera *camera_;
	GPContext *context_;
	bool opened_;
	size_t fail_count_;
	size_t timeout_;
};

} /* namespace gphoto */
} /* namespace yuri */
#endif /* GPHOTO_H_ */
