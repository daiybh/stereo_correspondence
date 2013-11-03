/*
 * UVSink.h
 *
 *  Created on: 23.10.2013
 *      Author: neneko
 */

#ifndef UVSINK_H_
#define UVSINK_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/VideoFrame.h"
#include "types.h"
//#include <unordered_set>

namespace yuri {
namespace ultragrid {
namespace detail {

typedef function<void* (char *, unsigned int)> 	display_init_t;
typedef function<void (void *)> 				display_run_t;
typedef function<void (void *)>					display_done_t;
typedef function<video_frame* (void *)>			display_getf_t;
typedef function<int (void *, video_frame *, int)>	display_putf_t;
typedef function<int (void *, video_desc)> 		display_reconfigure_t;
typedef function<int (void *, int, void *, size_t *)> display_get_property_t;

struct uv_display_params {
	std::string					name;
	display_init_t				init_func;// 	= nullptr;
	display_run_t 				run_func;// 	= nullptr;
	display_done_t				done_func;// 	= nullptr;
	display_getf_t				getf_func;//	= nullptr;
	display_putf_t				putf_func;//	= nullptr;
	display_reconfigure_t		reconfigure_func;// = nullptr;
	display_get_property_t		get_property_func;// = nullptr;
};

}

#define UV_SINK_DETAIL(name) \
	yuri::ultragrid::detail::uv_display_params{ \
	#name, \
	&display_ ## name ## _init, \
	&display_ ## name ## _run, \
	&display_ ## name ## _done, \
	&display_ ## name ## _getf, \
	&display_ ## name ## _putf, \
	&display_ ## name ## _reconfigure, \
	&display_ ## name ## _get_property, \
}

class UVVideoSink: public core::SpecializedIOFilter<core::VideoFrame>
{
public:
	//static core::Parameters configure();
	UVVideoSink(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_display_params sink_params);
	virtual ~UVVideoSink() noexcept;


protected:
	bool init_sink(const std::string& format, int flags);

	void* device_;
private:

	void run() override;
	virtual core::pFrame do_special_single_step(const core::pVideoFrame& frame) override;
	virtual void child_ends_hook(core::pwThreadBase child, int code, size_t remaining_child_count) override;

	video_desc last_desc_;
	detail::uv_display_params sink_params_;

};




}
}


#endif /* UVSINK_H_ */
