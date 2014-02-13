/*!
 * @file 		UVVideoSource.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		23.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef UVVIDEOSOURCE_H_
#define UVVIDEOSOURCE_H_

#include "yuri/core/thread/IOThread.h"
extern "C" {
#include "video_capture.h"
}
namespace yuri {
namespace ultragrid {

namespace detail {
typedef function<vidcap_type*()>						vidcap_probe_t;
typedef function<void*(char *fmt, unsigned int flags)>	vidcap_init_t;
typedef function<void(void *)>							vidcap_done_t;
typedef function<void(void *)>							vidcap_finish_t;
typedef function<video_frame*(void *, audio_frame **)>	vidcap_grab_t;

struct capture_params {
	std::string		name;
	vidcap_init_t	init_func;
	vidcap_done_t	done_func;
	vidcap_done_t	finish_func;
	vidcap_grab_t	grab_func;
};

#define UV_CAPTURE_DETAIL(name) \
	yuri::ultragrid::detail::capture_params {\
		#name, 	\
		vidcap_ ## name ## _init,	\
		vidcap_ ## name ## _done,	\
		vidcap_ ## name ## _finish,	\
		vidcap_ ## name ## _grab,	\
	}

}

class UVVideoSource: public core::IOThread
{
public:
	UVVideoSource(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::capture_params capt_params);
	virtual ~UVVideoSource() noexcept;
protected:
	bool init_capture(const std::string& params);

private:

	virtual void run() override;
	void* state_;
	detail::capture_params capt_params_;
};

}
}


#endif /* UVVIDEOSOURCE_H_ */
