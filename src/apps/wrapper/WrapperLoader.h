/*
 * FBWrapper.h
 *
 *  Created on: 24. 1. 2015
 *      Author: neneko
 */

#ifndef WRAPPERLOADER_H_
#define WRAPPERLOADER_H_

#include "yuri/core/thread/XmlBuilder.h"
#include "yuri/core/Module.h"
#include "yuri/gl/FBGrabber.h"

#include <dlfcn.h>
extern "C" {
	#include "GL/glx.h"
}

namespace yuri{
namespace wrapper {


class WrapperLoader: public core::IOThread {
public:
	WrapperLoader(log::Log& log_, const std::string& config_file, const std::string& node_name);
	virtual ~WrapperLoader() noexcept;
	void* get_func(const std::string& name);
	void init();

	// Wrappers for FBGrabber methods
	void set_viewport(geometry_t geometry);
	void pre_swap();
	void post_swap();

private:
	class xml_wrapper: public core::XmlBuilder {
	public:
		xml_wrapper(log::Log& log_, WrapperLoader& builder, const std::string& config_file):
			XmlBuilder(log_, {}, config_file, {}),
			w_builder_(builder)
		{

		}
//		bool push_external_frame(position_t index, core::pFrame frame)
//		{
//			return push_frame(index, frame);
//		}
	private:
		virtual void child_ends_hook(core::pwThreadBase /* child */, int code,
				size_t /* remaining_child_count */) override
		{
			w_builder_.child_ends(get_this_ptr(), code);
		}
		WrapperLoader& w_builder_;
	};

	std::shared_ptr<fb_grabber::FBGrabber> get_grabber();

	virtual void child_ends_hook(core::pwThreadBase child, int code,
					size_t remaining_child_count) override;
	std::string config_file_;
	std::string node_name_;
	std::mutex builder_mutex_;
	std::shared_ptr<xml_wrapper> graph_;
	std::shared_ptr<fb_grabber::FBGrabber> grabber_;
};

}
}



#endif /* WRAPPERLOADER_H_ */
