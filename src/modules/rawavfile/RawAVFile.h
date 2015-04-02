/*
 * RawAVFile.h
 *
 *  Created on: Feb 9, 2012
 *      Author: neneko
 */

#ifndef AVDEMUXER_H_
#define AVDEMUXER_H_
#include "yuri/libav/libav.h"
#include "yuri/core/thread/IOFilter.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/core/utils/managed_resource.h"
extern "C" {
	#include <libavformat/avformat.h>
}

#include <vector>

namespace yuri {
namespace rawavfile {



class RawAVFile: public core::IOThread, public event::BasicEventConsumer {
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	RawAVFile(const log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~RawAVFile() noexcept;
	virtual bool 		set_param(const core::Parameter &param);

	struct 				stream_detail_t;
private:

	virtual void 		run() override;
	virtual bool 		do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	bool				open_file(const std::string& filename);
	bool				push_ready_frames();
	bool				process_file_end();

	bool				process_undecoded_frame(index_t idx, const AVPacket& packet);
	bool				decode_video_frame(index_t idx, const AVPacket& packet, AVFrame* av_frame, bool& keep_packet);
	bool				decode_audio_frame(index_t idx, const AVPacket& packet, AVFrame* av_frame, bool& keep_packet);
	core::utils::managed_resource<AVFormatContext>
						fmtctx_;




	std::string 		filename_;
	std::string 		next_filename_;

	std::vector<stream_detail_t>
						video_streams_;
	std::vector<stream_detail_t>
						audio_streams_;

	format_t 			format_out_;
	format_t 			video_format_out_;
	bool				decode_;
	double 				fps_;
	std::vector<timestamp_t>
						next_times_;
	std::vector<core::pFrame>
						frames_;
	size_t 				max_video_streams_;
	size_t 				max_audio_streams_;

	bool 				loop_;
	bool				reset_;
};

} /* namespace video */
} /* namespace yuri */
#endif /* AVDEMUXER_H_ */
